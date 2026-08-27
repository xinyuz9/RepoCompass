"""Memory recall policy, freshness, provenance, and exclusion tests."""

import hashlib
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from repocompass.agent.context import (
    ContextMemoryFreshness,
    ContextMemoryScope,
    ContextSection,
    ContextSourceKind,
    VerificationStatus,
    create_context_item,
)
from repocompass.agent.nodes import create_recall_memory_node
from repocompass.agent.planning import InvestigationPlan, PlanStep
from repocompass.memory.contracts import (
    MemoryFactKind,
    MemoryVerificationStatus,
    MemoryWriteAuthority,
    create_episodic_memory,
    create_repository_memory,
    create_repository_memory_source,
)
from repocompass.memory.identity import resolve_memory_repository_identity
from repocompass.memory.retrieval import (
    build_memory_recall_query,
    recall_memory_context,
)
from repocompass.memory.store import open_memory_store
from repocompass.tools.gateway import ToolExecutionContext
from repocompass.agent.context import RepositoryLocation


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def _repository(root: Path) -> tuple[Path, ToolExecutionContext]:
    repository = root / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "tests@example.com")
    _git(repository, "config", "user.name", "RepoCompass Tests")
    source = repository / "auth.py"
    source.write_text(
        "def authenticate(request):\n    return request.user is not None\n",
        encoding="utf-8",
    )
    _git(repository, "add", "auth.py")
    _git(repository, "commit", "--quiet", "-m", "fixture")
    return repository, ToolExecutionContext(repository)


def _repository_memory(context: ToolExecutionContext):
    identity = resolve_memory_repository_identity(context)
    commit_sha = context.commit_sha
    assert commit_sha is not None
    source_path = context.repository_root / "auth.py"
    source = create_repository_memory_source(
        location=RepositoryLocation(
            repository_id=context.repository_id,
            commit_sha=commit_sha,
            path="auth.py",
            line_start=1,
            line_end=2,
            symbol="authenticate",
        ),
        source_digest=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )
    return create_repository_memory(
        identity=identity,
        commit_sha=commit_sha,
        kind=MemoryFactKind.ENTRY_POINT,
        title="Authentication entry point",
        content="auth.authenticate validates the request user.",
        verification_status=MemoryVerificationStatus.VERIFIED,
        write_authority=MemoryWriteAuthority.VALIDATED_EVIDENCE,
        sources=(source,),
        tags=("authentication", "entry-point"),
        created_at_ms=1_000,
        verified_at_ms=1_000,
    )


def _episode(context: ToolExecutionContext):
    identity = resolve_memory_repository_identity(context)
    return create_episodic_memory(
        identity=identity,
        commit_sha=context.commit_sha,
        run_id="episode-run",
        thread_id="episode-thread",
        turn_number=1,
        termination_status="accepted",
        objective_digest=hashlib.sha256(b"authentication objective").hexdigest(),
        summary="Authentication investigation found auth.py.",
        query_sequence=("authentication", "authenticate"),
        successful_paths=("auth.py",),
        failed_hypotheses=("No separate middleware module was found.",),
        final_result_digest=hashlib.sha256(b"final result").hexdigest(),
        created_at_ms=1_100,
    )


def _step() -> PlanStep:
    return PlanStep(
        id="inspect-auth",
        objective="Find the authentication entry point",
        expected_evidence=["authentication authenticate auth.py"],
        status="active",
    )


def test_recall_exposes_current_fact_and_episode_only_as_non_evidence(
    tmp_path: Path,
) -> None:
    _, context = _repository(tmp_path)
    memory = _repository_memory(context)
    episode = _episode(context)
    with open_memory_store(tmp_path / "memory.sqlite") as store:
        store.put_repository_memory(memory, context=context, run_id="write-run")
        store.put_episodic_memory(episode, context=context)

        result = recall_memory_context(
            store,
            repo_path=context.repository_root,
            objective="Explain authentication.",
            active_step=_step(),
            audit_at_ms=2_000,
            run_id="recall-run",
        )

    assert result.report.repository_result_count == 1
    assert result.report.episodic_result_count == 1
    assert len(result.context_items) == 2
    repository_item, episode_item = result.context_items
    assert repository_item.section is ContextSection.MEMORY
    assert repository_item.source.kind is ContextSourceKind.MEMORY
    assert repository_item.source.raw_artifact_id is None
    assert repository_item.source.memory is not None
    assert repository_item.source.memory.scope is ContextMemoryScope.REPOSITORY
    assert repository_item.source.memory.freshness is ContextMemoryFreshness.CURRENT
    assert repository_item.verification_status is VerificationStatus.INFERRED
    assert "lead_only_revalidate_with_repository_tools" in repository_item.content
    assert episode_item.source.memory is not None
    assert episode_item.source.memory.scope is ContextMemoryScope.EPISODIC
    assert episode_item.source.memory.freshness is ContextMemoryFreshness.OBSERVED
    assert episode_item.verification_status is VerificationStatus.UNVERIFIED
    assert "never_code_evidence" in episode_item.content

    report_json = result.report.model_dump_json()
    assert memory.content not in report_json
    assert episode.summary not in report_json
    assert result.report.context_item_ids == tuple(
        item.item_id for item in result.context_items
    )


def test_changed_source_is_reported_but_excluded_from_context(tmp_path: Path) -> None:
    repository, context = _repository(tmp_path)
    memory = _repository_memory(context)
    with open_memory_store(tmp_path / "memory.sqlite") as store:
        store.put_repository_memory(memory, context=context)
        (repository / "auth.py").write_text("changed = True\n", encoding="utf-8")

        result = recall_memory_context(
            store,
            repo_path=repository,
            objective="Find authentication.",
            active_step=_step(),
            audit_at_ms=2_000,
        )

    assert result.report.repository_result_count == 1
    assert result.context_items == ()
    decision = result.report.decisions[0]
    assert decision.freshness is ContextMemoryFreshness.STALE
    assert decision.context_item_id is None
    assert decision.staleness_reason is not None


def test_new_commit_is_downgraded_to_unverified_revalidation_lead(
    tmp_path: Path,
) -> None:
    repository, context = _repository(tmp_path)
    memory = _repository_memory(context)
    with open_memory_store(tmp_path / "memory.sqlite") as store:
        store.put_repository_memory(memory, context=context)
        (repository / "auth.py").write_text(
            "def authenticate(request):\n    return bool(request.user)\n",
            encoding="utf-8",
        )
        _git(repository, "add", "auth.py")
        _git(repository, "commit", "--quiet", "-m", "change auth")

        result = recall_memory_context(
            store,
            repo_path=repository,
            objective="Find authentication.",
            active_step=_step(),
            audit_at_ms=2_000,
        )

    assert len(result.context_items) == 1
    item = result.context_items[0]
    assert item.verification_status is VerificationStatus.UNVERIFIED
    assert item.source.memory is not None
    assert item.source.memory.freshness is ContextMemoryFreshness.REVALIDATION_REQUIRED
    assert item.source.memory.staleness_reason is not None


def test_memory_source_cannot_be_promoted_to_verified_context(
    tmp_path: Path,
) -> None:
    _, context = _repository(tmp_path)
    memory = _repository_memory(context)
    with open_memory_store(tmp_path / "memory.sqlite") as store:
        store.put_repository_memory(memory, context=context)
        item = recall_memory_context(
            store,
            repo_path=context.repository_root,
            objective="Find authentication.",
            active_step=_step(),
            audit_at_ms=2_000,
        ).context_items[0]

    with pytest.raises(ValidationError, match="verified context"):
        create_context_item(
            source=item.source,
            section=item.section,
            verification_status=VerificationStatus.VERIFIED,
            content=item.content,
            estimated_tokens=1,
            priority=item.priority,
        )


def test_query_is_deterministic_bounded_and_safe_for_long_unicode_tokens() -> None:
    step = PlanStep(
        id="unicode",
        objective="界" * 100,
        expected_evidence=[f"token-{index}" for index in range(30)],
        status="active",
    )

    first = build_memory_recall_query("认证 authentication", step)
    second = build_memory_recall_query("认证 authentication", step)
    tokens = first.split()

    assert first == second
    assert 1 <= len(tokens) <= 16
    assert all(len(token) <= 64 for token in tokens)
    assert "\x00" not in first


def test_recall_rejects_out_of_policy_limits_before_query(tmp_path: Path) -> None:
    _, context = _repository(tmp_path)
    with open_memory_store(tmp_path / "memory.sqlite") as store:
        with pytest.raises(ValueError, match="between 1 and 10"):
            recall_memory_context(
                store,
                repo_path=context.repository_root,
                objective="Find authentication.",
                active_step=_step(),
                max_repository_results=11,
                audit_at_ms=2_000,
            )


def test_disabled_recall_performs_zero_store_and_clock_calls(tmp_path: Path) -> None:
    def fail_clock() -> int:
        raise AssertionError("disabled recall must not read the clock")

    node = create_recall_memory_node(object(), fail_clock)  # type: ignore[arg-type]
    update = node(
        {
            "repo_path": tmp_path,
            "objective": "Inspect authentication.",
            "investigation_plan": None,
            "memory_retrieval_enabled": False,
            "max_repository_memory_results": 5,
            "max_episodic_memory_results": 5,
        }
    )

    assert update == {
        "memory_recall_report": None,
        "memory_context_items": [],
    }


def test_enabled_recall_fails_closed_without_store(tmp_path: Path) -> None:
    plan = InvestigationPlan(steps=[_step()])
    node = create_recall_memory_node(None, lambda: 2_000)

    with pytest.raises(ValueError, match="without a memory store"):
        node(
            {
                "repo_path": tmp_path,
                "objective": "Inspect authentication.",
                "investigation_plan": plan,
                "memory_retrieval_enabled": True,
                "max_repository_memory_results": 5,
                "max_episodic_memory_results": 5,
            }
        )
