"""Ephemeral sanitized dataset for a credential-free dashboard demonstration."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.runnables import RunnableLambda

from repocompass.agent.candidate import (
    CandidateRelation,
    CandidateResult,
    EvidenceItem,
)
from repocompass.agent.candidate_output import EvidenceOnlyCandidateResult
from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.context import RepositoryLocation
from repocompass.agent.cost import (
    ModelPrice,
    PricingTable,
    create_pricing_table,
)
from repocompass.agent.evaluation import SemanticAssessment
from repocompass.agent.lifecycle import ContinueRequest, HumanReviewPolicy
from repocompass.agent.planning import InvestigationPlan, PlanStep, PlanProposal
from repocompass.agent.runner import AgentRunner, RunRequest
from repocompass.agent.trace_sink import JsonlTraceSink
from repocompass.memory.contracts import (
    MemoryFactKind,
    MemoryVerificationStatus,
    MemoryWriteAuthority,
    create_repository_memory,
    create_repository_memory_source,
)
from repocompass.memory.identity import resolve_memory_repository_identity
from repocompass.memory.store import MemoryStore, open_memory_store
from repocompass.models import WorkflowModels
from repocompass.storage.raw_store import open_raw_run_store
from repocompass.structure import EvidenceRelationKind
from repocompass.tools.gateway import ToolExecutionContext

DEMO_THREAD_ID = "demo-proposal-gate"
DEMO_RUN_ID = "demo-run-001"
DEMO_OBJECTIVE = "Trace how proposal verification gates application and identify the regression test."
DEMO_MODEL = "deterministic-fixture-v1"
DEMO_PROVIDER = "demo"
DEMO_CREATED_AT_MS = 1_785_542_400_000

_POLICY_SOURCE = '''"""Proposal verification policy used by the sanitized demo."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Proposal:
    target: str
    replacement: str


class ProposalVerifier:
    """Validate a bounded proposal before application."""

    def verify(self, proposal: Proposal) -> bool:
        has_content = bool(proposal.replacement.strip())
        stays_local = ".." not in proposal.target.split("/")
        return has_content and stays_local
'''

_API_SOURCE = '''"""Application boundary for verified proposals."""

from .policy import Proposal, ProposalVerifier


def apply_proposal(
    proposal: Proposal,
    workspace: dict[str, str],
    verifier: ProposalVerifier,
) -> None:
    """Apply only a proposal accepted by the verifier."""

    if not verifier.verify(proposal):
        raise ValueError("proposal rejected")
    workspace[proposal.target] = proposal.replacement
'''

_TEST_SOURCE = """from pkg.api import apply_proposal
from pkg.policy import Proposal, ProposalVerifier


def test_apply_proposal_rejects_parent_escape() -> None:
    workspace: dict[str, str] = {}
    proposal = Proposal(target="../secret.py", replacement="blocked = True")

    try:
        apply_proposal(proposal, workspace, ProposalVerifier())
    except ValueError:
        pass

    assert workspace == {}
"""


class DashboardDemoError(RuntimeError):
    """A sanitized demo workspace could not be created safely."""


@dataclass(frozen=True, slots=True)
class DashboardDemoWorkspace:
    """Paths and immutable pricing required to serve one ephemeral demo."""

    root: Path
    repository_path: Path
    checkpoint_database: Path
    raw_store_database: Path
    memory_database: Path
    trace_directory: Path
    pricing_table: PricingTable
    thread_id: str = DEMO_THREAD_ID
    run_id: str = DEMO_RUN_ID


def create_dashboard_demo(
    destination: Path, *, turns: int = 1
) -> DashboardDemoWorkspace:
    """Generate bounded production-format turns without provider credentials."""

    if type(turns) is not int or not 1 <= turns <= 3:
        raise DashboardDemoError("demo turns must be between 1 and 3")
    root = _prepare_empty_destination(destination)
    repository = root / "proposal-gate-demo"
    state = root / "state"
    checkpoint_database = state / "checkpoints.sqlite"
    raw_store_database = state / "raw_runs.sqlite"
    memory_database = state / "memory.sqlite"
    trace_directory = state / "traces"
    _write_demo_repository(repository)
    context = ToolExecutionContext(repository)
    pricing_table = _demo_pricing_table()

    with (
        open_sqlite_checkpointer(checkpoint_database) as checkpointer,
        open_raw_run_store(raw_store_database) as raw_store,
        open_memory_store(memory_database) as memory_store,
    ):
        _seed_repository_memory(memory_store, context=context)
        runner = AgentRunner(
            _demo_models(),
            checkpointer=checkpointer,
            raw_run_store=raw_store,
            pricing_table=pricing_table,
            memory_store=memory_store,
            episode_clock_factory=lambda: DEMO_CREATED_AT_MS + 1_000,
            run_id_factory=lambda: DEMO_RUN_ID,
            thread_id_factory=lambda: DEMO_THREAD_ID,
        )
        outcome = runner.run(
            RunRequest(
                thread_id=DEMO_THREAD_ID,
                repo_path=repository,
                objective=DEMO_OBJECTIVE,
                max_tool_calls_per_step=2,
                max_explorer_calls_per_step=3,
                max_no_progress_rounds=2,
                max_plan_revisions=2,
                # Keep the import record whole alongside the final source excerpts.
                # This scripted UI fixture does not change paid-pilot admission caps.
                context_budget_tokens=8_192,
                human_review_policy=HumanReviewPolicy.NEVER,
                memory_retrieval_enabled=True,
                max_repository_memory_results=3,
                max_episodic_memory_results=3,
            )
        )
        if (
            outcome.final_result is None
            or outcome.final_result.structure is None
            or outcome.usage_cost_report is None
            or not raw_store.list_context_packs(DEMO_RUN_ID)
            or not raw_store.list_model_invocations(DEMO_RUN_ID)
            or not memory_store.list_memory_metadata(context=context, max_results=10)
        ):
            raise DashboardDemoError("demo generation produced an incomplete run")
        JsonlTraceSink().write(
            outcome.trace,
            trace_directory / f"{DEMO_RUN_ID}.jsonl",
        )
        for turn in range(2, turns + 1):
            follow_run_id = f"demo-run-{turn:03d}"
            follow_runner = AgentRunner(
                _demo_models(),
                checkpointer=checkpointer,
                raw_run_store=raw_store,
                pricing_table=pricing_table,
                memory_store=memory_store,
                episode_clock_factory=lambda: DEMO_CREATED_AT_MS + 2_000,
                run_id_factory=lambda run_id=follow_run_id: run_id,
            )
            follow = follow_runner.continue_thread(
                ContinueRequest(
                    thread_id=DEMO_THREAD_ID,
                    prompt="Review the proposal verification path again and retain earlier findings.",
                )
            )
            if follow.final_result is None:
                raise DashboardDemoError("demo continuation produced no final answer")
            JsonlTraceSink().write(
                follow.trace, trace_directory / f"{follow_run_id}.jsonl"
            )

    return DashboardDemoWorkspace(
        root=root,
        repository_path=repository,
        checkpoint_database=checkpoint_database,
        raw_store_database=raw_store_database,
        memory_database=memory_database,
        trace_directory=trace_directory,
        pricing_table=pricing_table,
    )


def _prepare_empty_destination(destination: Path) -> Path:
    if not isinstance(destination, Path):
        raise TypeError("destination must be a pathlib.Path")
    root = destination.expanduser().resolve(strict=False)
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise DashboardDemoError("demo destination must be an empty directory")
        try:
            has_entries = next(root.iterdir(), None) is not None
        except OSError as error:
            raise DashboardDemoError(
                "demo destination could not be inspected"
            ) from error
        if has_entries:
            raise DashboardDemoError("demo destination must be empty")
    else:
        try:
            root.mkdir(parents=True, mode=0o700)
        except OSError as error:
            raise DashboardDemoError("demo destination could not be created") from error
    return root.resolve(strict=True)


def _write_demo_repository(repository: Path) -> None:
    try:
        (repository / "pkg").mkdir(parents=True)
        (repository / "tests").mkdir()
        (repository / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        (repository / "pkg" / "policy.py").write_text(
            _POLICY_SOURCE,
            encoding="utf-8",
        )
        (repository / "pkg" / "api.py").write_text(_API_SOURCE, encoding="utf-8")
        (repository / "tests" / "test_api.py").write_text(
            _TEST_SOURCE,
            encoding="utf-8",
        )
        (repository / "README.md").write_text(
            "# Proposal Gate Demo\n\nSanitized local fixture for RepoCompass.\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise DashboardDemoError("demo repository could not be written") from error

    commands = (
        ("init", "--quiet"),
        ("config", "user.email", "demo@repocompass.local"),
        ("config", "user.name", "RepoCompass Demo"),
        ("add", "README.md", "pkg", "tests"),
    )
    try:
        for arguments in commands:
            _run_git(repository, arguments)
        commit_environment = dict(os.environ)
        commit_environment.update(
            {
                "GIT_AUTHOR_DATE": "2026-08-01T00:00:00Z",
                "GIT_COMMITTER_DATE": "2026-08-01T00:00:00Z",
            }
        )
        _run_git(
            repository,
            ("commit", "--quiet", "-m", "sanitized proposal-gate fixture"),
            environment=commit_environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise DashboardDemoError("demo Git snapshot could not be created") from error


def _run_git(
    repository: Path,
    arguments: tuple[str, ...],
    *,
    environment: dict[str, str] | None = None,
) -> None:
    subprocess.run(
        ("git", *arguments),
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
        shell=False,
    )


def _seed_repository_memory(
    memory_store: MemoryStore, *, context: ToolExecutionContext
) -> None:
    commit_sha = context.commit_sha
    if commit_sha is None:
        raise DashboardDemoError("demo repository did not produce a commit")
    policy_path = context.repository_root / "pkg" / "policy.py"
    source = create_repository_memory_source(
        location=RepositoryLocation(
            repository_id=context.repository_id,
            commit_sha=commit_sha,
            path="pkg/policy.py",
            line_start=12,
            line_end=18,
            symbol="pkg.policy.ProposalVerifier",
        ),
        source_digest=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        evidence_claim_digest=hashlib.sha256(
            b"ProposalVerifier rejects empty content and parent escapes."
        ).hexdigest(),
    )
    memory = create_repository_memory(
        identity=resolve_memory_repository_identity(context),
        commit_sha=commit_sha,
        kind=MemoryFactKind.ARCHITECTURE_FACT,
        title="Proposal application safety invariant",
        content=(
            "ProposalVerifier must accept a proposal before apply_proposal mutates "
            "the workspace."
        ),
        verification_status=MemoryVerificationStatus.VERIFIED,
        write_authority=MemoryWriteAuthority.VALIDATED_EVIDENCE,
        sources=(source,),
        tags=("proposal", "verification"),
        created_at_ms=DEMO_CREATED_AT_MS,
        verified_at_ms=DEMO_CREATED_AT_MS,
    )
    memory_store.put_repository_memory(
        memory,
        context=context,
        run_id="demo-memory-seed",
    )


def _demo_models() -> WorkflowModels:
    explorer = _DemoExplorer()
    planner = _DemoPlanner()
    evaluator = _DemoEvaluator()
    return WorkflowModels(
        explorer=cast(BaseChatModel, explorer),
        planner=cast(BaseChatModel, planner),
        evaluator=cast(BaseChatModel, evaluator),
    )


class _DemoModelBase:
    metadata = {
        "repocompass_provider": DEMO_PROVIDER,
        "repocompass_model": DEMO_MODEL,
    }


class _DemoPlanner(_DemoModelBase):
    def with_structured_output(self, schema: object, *_: object, **__: object):
        if schema is not PlanProposal:
            raise DashboardDemoError("demo planner received an unexpected schema")
        return RunnableLambda(lambda _: _envelope(_demo_plan(), 48, 18))


class _DemoEvaluator(_DemoModelBase):
    def with_structured_output(self, schema: object, *_: object, **__: object):
        if schema is not SemanticAssessment:
            raise DashboardDemoError("demo evaluator received an unexpected schema")
        return RunnableLambda(lambda _: _envelope(_demo_assessment(), 82, 24))


class _DemoExplorer(_DemoModelBase):
    def __init__(self) -> None:
        self._response_index = 0
        self._candidate_index = 0

    def bind_tools(self, *_: object, **__: object):
        return RunnableLambda(self._next_response)

    def with_structured_output(self, schema: object, *_: object, **__: object):
        if schema not in (CandidateResult, EvidenceOnlyCandidateResult):
            raise DashboardDemoError("demo explorer received an unexpected schema")
        return RunnableLambda(lambda messages: self._next_candidate(messages, schema))

    def _next_response(self, _: object) -> AIMessage:
        responses = _explorer_responses()
        if self._response_index >= len(responses):
            raise DashboardDemoError("demo explorer response script was exhausted")
        response = responses[self._response_index]
        self._response_index += 1
        return response

    def _next_candidate(self, _: object, schema: object) -> dict[str, object]:
        candidates = _demo_candidates()
        if self._candidate_index >= len(candidates):
            raise DashboardDemoError("demo candidate script was exhausted")
        candidate = candidates[self._candidate_index]
        self._candidate_index += 1
        if schema is EvidenceOnlyCandidateResult:
            if candidate.relations:
                raise DashboardDemoError("demo relation requires selected AST evidence")
            return _envelope(
                EvidenceOnlyCandidateResult.model_validate_json(
                    candidate.model_dump_json(exclude={"relations"})
                ),
                96,
                36,
            )
        return _envelope(candidate, 96, 36)


def _demo_plan() -> InvestigationPlan:
    return InvestigationPlan(
        steps=[
            PlanStep(
                id="locate-application",
                objective="Locate the proposal application entry point.",
                expected_evidence=["A repository-relative application function."],
                status="active",
            ),
            PlanStep(
                id="verify-gate",
                objective="Verify the policy dependency that gates mutation.",
                expected_evidence=["A structurally verified module relation."],
                status="pending",
            ),
            PlanStep(
                id="confirm-regression",
                objective="Identify a regression test for rejected proposals.",
                expected_evidence=["A bounded test source location."],
                status="pending",
            ),
        ]
    )


def _demo_candidates() -> tuple[CandidateResult, ...]:
    evidence = (
        EvidenceItem(
            path="pkg/api.py",
            line_start=3,
            line_end=15,
            symbol="pkg.api",
            claim="apply_proposal checks the verifier before mutating the workspace.",
        ),
        EvidenceItem(
            path="pkg/policy.py",
            line_start=12,
            line_end=18,
            symbol="pkg.policy.ProposalVerifier",
            claim="ProposalVerifier rejects empty replacements and parent escapes.",
        ),
        EvidenceItem(
            path="tests/test_api.py",
            line_start=1,
            line_end=14,
            symbol="tests.test_api",
            claim="The regression test proves a rejected proposal leaves state unchanged.",
        ),
    )
    relation = CandidateRelation(
        source="pkg.api",
        relation=EvidenceRelationKind.IMPORTS,
        target="pkg.policy",
        supporting_evidence_indices=(0, 1),
        explanation="The application module imports its policy dependency directly.",
    )
    return (
        CandidateResult(
            evidence=[evidence[0].model_copy(update={"symbol": None})],
            summary="The application entry point validates before mutation.",
            unresolved_questions=[
                "The verifier policy and regression test remain to inspect."
            ],
        ),
        CandidateResult(
            evidence=[evidence[0], evidence[1]],
            relations=[relation],
            summary="Proposal application is gated by the imported verifier policy.",
            unresolved_questions=["The rejection regression test remains to inspect."],
        ),
        CandidateResult(
            evidence=list(evidence),
            relations=[relation],
            summary=(
                "apply_proposal calls ProposalVerifier before mutation; invalid parent "
                "escapes are rejected and the regression test confirms no write occurs."
            ),
            unresolved_questions=[],
        ),
    )


def _demo_assessment() -> SemanticAssessment:
    return SemanticAssessment(
        active_step_completed=True,
        answers_objective=True,
        evidence_supports_summary=True,
        has_critical_omissions=False,
        missing_evidence=[],
        recommended_focus=[],
        summary="The bounded evidence completes the active investigation step.",
    )


def _explorer_responses() -> tuple[AIMessage, ...]:
    return (
        _tool_message(
            "list_tree",
            {"relative_path": ".", "max_depth": 3, "max_entries": 40},
            "demo-tree",
        ),
        _tool_message(
            "search_text",
            {"query": "verify", "globs": ["*.py"], "max_results": 20},
            "demo-search",
        ),
        _text_message("The application entry point is bounded and located."),
        _tool_message(
            "inspect_imports",
            {"relative_path": "pkg/api.py", "max_results": 10},
            "demo-inspect-api",
        ),
        _tool_message(
            "read_code",
            {"relative_path": "pkg/policy.py", "start_line": 12, "end_line": 18},
            "demo-read-policy",
        ),
        _text_message("The application imports and invokes the verifier gate."),
        _tool_message(
            "read_code",
            {"relative_path": "tests/test_api.py", "start_line": 1, "end_line": 14},
            "demo-read-test",
        ),
        _text_message("The regression test confirms rejected proposals do not write."),
    )


def _tool_message(
    name: str,
    arguments: dict[str, object],
    call_id: str,
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name=name,
                args=arguments,
                id=call_id,
                type="tool_call",
            )
        ],
        usage_metadata={"input_tokens": 54, "output_tokens": 12, "total_tokens": 66},
    )


def _text_message(content: str) -> AIMessage:
    return AIMessage(
        content=content,
        usage_metadata={"input_tokens": 72, "output_tokens": 18, "total_tokens": 90},
    )


def _envelope(
    parsed: object, input_tokens: int, output_tokens: int
) -> dict[str, object]:
    return {
        "raw": AIMessage(
            content=f"sanitized {type(parsed).__name__} response",
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        ),
        "parsed": parsed,
        "parsing_error": None,
    }


def _demo_pricing_table() -> PricingTable:
    return create_pricing_table(
        version="demo-2026-08",
        effective_from="2026-08-01",
        entries=(
            ModelPrice(
                provider=DEMO_PROVIDER,
                model=DEMO_MODEL,
                input_usd_per_million_tokens="0.5",
                output_usd_per_million_tokens="1.5",
            ),
        ),
    )
