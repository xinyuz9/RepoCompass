"""No remaining-work shortcut without complete, current, visible evidence links."""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.candidate_output import EvidenceOnlyCandidateResult
from repocompass.agent.checkpoint import RepoCompassCheckpointSerializer
from repocompass.agent.context_builder import ContextBudgetPolicy, build_active_context
from repocompass.agent.context_gathering import gather_evaluator_context
from repocompass.agent.evaluation import SemanticAssessment, validate_candidate
from repocompass.agent.nodes import create_evaluate_candidate_node, decide_next_action
from repocompass.agent.plan_coverage import (
    assess_plan_coverage,
    selected_evidence_indices,
    StepCoverage,
)
from repocompass.agent.planning import InvestigationPlan, PlanStep, PlanProposal


def _plan():
    return InvestigationPlan(
        steps=[
            PlanStep(
                id="inspect",
                objective="Inspect the flag.",
                expected_evidence=["Flag source."],
                status="active",
            ),
            PlanStep(
                id="summarize",
                objective="Summarize the flag.",
                expected_evidence=["Flag declaration."],
                status="pending",
            ),
        ]
    )


def _candidate():
    return CandidateResult(
        evidence=[EvidenceItem(path="config.py", line_start=1, claim="FLAG is True.")],
        summary="The flag is True.",
        unresolved_questions=[],
    )


def _claim():
    return {
        "step_id": "summarize",
        "disposition": "covered",
        "objective_evidence_indices": [0],
        "requirements": [
            {
                "requirement_index": 0,
                "evidence_indices": [0],
                "explanation": "The source assigns True.",
            }
        ],
        "explanation": "The existing answer states the verified flag value.",
    }


def _assessment(claims=None):
    return SemanticAssessment(
        active_step_completed=True,
        answers_objective=True,
        evidence_supports_summary=True,
        has_critical_omissions=False,
        missing_evidence=[],
        recommended_focus=[],
        summary="The source supports the answer.",
        remaining_step_coverage=[
            StepCoverage.model_validate(c)
            for c in (claims if claims is not None else [_claim()])
        ],
    )


def _state(path):
    return {
        "repo_path": path,
        "objective": "Explain the flag.",
        "investigation_plan": _plan(),
        "candidate_result": _candidate(),
        "context_budget_tokens": 16_384,
        "context_build_count": 0,
        "step_stop_reason": None,
        "step_explorer_call_count": 1,
        "max_explorer_calls_per_step": 2,
        "plan_revision_count": 0,
        "max_plan_revisions": 1,
    }


def _evaluate(path, *, assessment=None, state=None):
    (path / "config.py").write_text("FLAG = True\n")
    model = MagicMock(spec=BaseChatModel)
    seen = []

    def response(messages):
        seen.extend(messages)
        return assessment or _assessment()

    model.with_structured_output.return_value = RunnableLambda(response)
    state = state or _state(path)
    update = create_evaluate_candidate_node(model)(state)
    return state | update, seen


@pytest.mark.parametrize("disposition", ["covered", "unresolved"])
def test_empty_requirement_evidence_is_parseable_but_never_approved(disposition):
    claim = _claim()
    claim["disposition"] = disposition
    claim["requirements"][0]["evidence_indices"] = []
    assessment = _assessment([claim])
    decision = assess_plan_coverage(
        objective="Explain the flag.",
        plan=_plan(),
        candidate=_candidate(),
        claims=assessment.remaining_step_coverage,
        selected_indices=frozenset({0}),
        evaluation_passed=True,
    )
    assert not decision.approved_step_ids
    assert decision.reason in {"incomplete", "invalid_references"}


@pytest.mark.parametrize("disposition", ["covered", "unresolved"])
@pytest.mark.parametrize("missing", [True, False])
def test_missing_diagnostic_explanation_never_approves_work(disposition, missing):
    claim = _claim()
    claim["disposition"] = disposition
    if missing:
        del claim["explanation"]
    else:
        claim["explanation"] = None
    assessment = _assessment([claim])
    decision = assess_plan_coverage(
        objective="Explain the flag.",
        plan=_plan(),
        candidate=_candidate(),
        claims=assessment.remaining_step_coverage,
        selected_indices=frozenset({0}),
        evaluation_passed=True,
    )
    assert decision.reason == "incomplete"
    assert not decision.approved_step_ids


def test_incomplete_diagnostic_preserves_negative_review_in_real_node(tmp_path):
    claim = _claim()
    del claim["explanation"]
    claim["disposition"] = "unresolved"
    assessment = _assessment([claim]).model_copy(update={"answers_objective": False})
    state, _ = _evaluate(tmp_path, assessment=assessment)
    assert not state["evaluation_report"].passed
    assert decide_next_action(state)["next_action"] == "advance_step"


def test_real_evaluator_supplies_full_plan_and_controller_accepts_covered_work(
    tmp_path,
):
    state, seen = _evaluate(tmp_path)
    report = state["evaluation_report"]
    assert report.plan_coverage.reason == "all_pending_covered"
    assert report.plan_coverage.approved_step_ids == ["summarize"]
    assert "Full bounded investigation plan" in seen[1].content
    assert "Summarize the flag." in seen[1].content
    assert decide_next_action(state) == {"next_action": "accepted"}
    # Never relabel an unexecuted step's historical status.
    assert state["investigation_plan"] == _plan()
    serializer = RepoCompassCheckpointSerializer()
    for _ in range(2):
        report = serializer.loads_typed(serializer.dumps_typed(report))
    assert report == state["evaluation_report"]
    assert (
        decide_next_action(state | {"evaluation_report": report})["next_action"]
        == "accepted"
    )


@pytest.mark.parametrize(
    "mutation", ["objective", "candidate", "plan", "missing_candidate"]
)
def test_coverage_is_bound_to_current_identity(tmp_path, mutation):
    state, _ = _evaluate(tmp_path)
    if mutation == "objective":
        state["objective"] = "A new task."
    elif mutation == "candidate":
        state["candidate_result"] = _candidate().model_copy(
            update={"summary": "A changed answer."}
        )
    elif mutation == "missing_candidate":
        state["candidate_result"] = None
    else:
        changed = _plan().model_dump(mode="json")
        changed["steps"][1]["expected_evidence"].append("An additional required test.")
        state["investigation_plan"] = InvestigationPlan.model_validate(changed)
    assert decide_next_action(state)["next_action"] == "advance_step"


@pytest.mark.parametrize(
    "mutation",
    [
        "absent",
        "unknown_step",
        "duplicate_step",
        "unresolved",
        "missing_requirement",
        "wrong_requirement",
        "duplicate_requirement",
        "missing_objective",
        "out_of_range",
        "duplicate_reference",
        "excluded_reference",
    ],
)
def test_incomplete_or_unseen_coverage_cannot_bypass_pending_work(mutation):
    claim = _claim()
    claims = [claim]
    selected = frozenset({0})
    if mutation == "absent":
        claims = []
    elif mutation == "unknown_step":
        claim["step_id"] = "made-up"
    elif mutation == "duplicate_step":
        claims.append(deepcopy(claim))
    elif mutation == "unresolved":
        claim["disposition"] = "unresolved"
    elif mutation == "missing_requirement":
        claim["requirements"] = []
    elif mutation == "wrong_requirement":
        claim["requirements"][0]["requirement_index"] = 1
    elif mutation == "duplicate_requirement":
        claim["requirements"].append(deepcopy(claim["requirements"][0]))
    elif mutation == "missing_objective":
        claim["objective_evidence_indices"] = []
    elif mutation == "out_of_range":
        claim["requirements"][0]["evidence_indices"] = [99]
    elif mutation == "duplicate_reference":
        claim["objective_evidence_indices"] = [0, 0]
    else:
        selected = frozenset()
    decision = assess_plan_coverage(
        objective="Explain the flag.",
        plan=_plan(),
        candidate=_candidate(),
        claims=[StepCoverage.model_validate(c) for c in claims],
        selected_indices=selected,
        evaluation_passed=True,
    )
    assert not decision.approved_step_ids


def test_positive_semantic_review_cannot_override_deterministic_defects(tmp_path):
    state = _state(tmp_path)
    state["candidate_result"] = CandidateResult(
        evidence=[EvidenceItem(path="missing.py", line_start=1, claim="Not real.")],
        summary="Not real.",
        unresolved_questions=[],
    )
    state, _ = _evaluate(tmp_path, state=state)
    assert not state["evaluation_report"].passed
    assert state["evaluation_report"].plan_coverage.reason == "evaluation_failed"
    assert decide_next_action(state)["next_action"] == "continue_step"


def test_no_coverage_keeps_old_conservative_controller_behavior(tmp_path):
    state, _ = _evaluate(tmp_path, assessment=_assessment([]))
    assert state["evaluation_report"].passed
    assert decide_next_action(state)["next_action"] == "advance_step"


def test_context_exclusion_removes_coverage_authority(tmp_path):
    (tmp_path / "config.py").write_text("FLAG = True\n")
    evidence = validate_candidate(tmp_path, _candidate()).evidence_pack
    gathered = gather_evaluator_context(
        repo_path=tmp_path,
        objective="Explain the flag.",
        active_step=_plan().active_step,
        investigation_plan=_plan(),
        candidate_result=_candidate(),
        deterministic_issues=(),
        evidence_pack=evidence,
        budget_tokens=16_384,
    )
    assert selected_evidence_indices(gathered.pack, evidence) == frozenset({0})
    budget = sum(item.estimated_tokens for item in gathered.pack.items if item.pinned)
    excluded = build_active_context(
        candidates=gathered.pack.items, policy=ContextBudgetPolicy(budget_tokens=budget)
    )
    assert selected_evidence_indices(excluded, evidence) == frozenset()


@pytest.mark.parametrize("symbol", ["test", "pkg.FLAG", "authenticate"])
def test_source_observation_schema_rejects_unbound_ast_selectors(symbol):
    payload = _candidate().model_dump(mode="json", exclude={"relations"})
    payload["evidence"][0]["symbol"] = symbol
    with pytest.raises(ValueError):
        EvidenceOnlyCandidateResult.model_validate(payload)


def test_covered_plan_finalizes_runner_memory_without_extra_step_calls(tmp_path):
    import subprocess
    from langchain_core.messages import AIMessage
    from repocompass.agent.checkpoint import open_sqlite_checkpointer
    from repocompass.agent.runner import AgentRunner, RunRequest, ResumeRequest
    from repocompass.evaluation.live_cli import prewarm_readiness
    from repocompass.memory.store import open_memory_store
    from repocompass.models import WorkflowModels

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "config.py").write_text("FLAG = True\n")
    for args in (
        ("init", "-q"),
        ("config", "user.email", "fixture@example.com"),
        ("config", "user.name", "Fixture"),
        ("add", "config.py"),
        ("commit", "-qm", "fixture"),
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    calls = []
    responses = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_code",
                        "args": {
                            "relative_path": "config.py",
                            "start_line": 1,
                            "end_line": 1,
                        },
                        "id": "read",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Enough source evidence."),
        ]
    )
    model = MagicMock(spec=BaseChatModel)

    def explore(_):
        calls.append("explorer")
        return next(responses)

    def structured(schema, **_):
        def invoke(_):
            calls.append(schema.__name__)
            if schema is PlanProposal:
                return PlanProposal.model_validate(
                    _plan().model_dump(exclude={"steps": {"__all__": {"status"}}})
                )
            if schema is EvidenceOnlyCandidateResult:
                return EvidenceOnlyCandidateResult.model_validate_json(
                    _candidate().model_dump_json(exclude={"relations"})
                )
            assert schema is SemanticAssessment
            return _assessment()

        return RunnableLambda(invoke)

    model.bind_tools.return_value = RunnableLambda(explore)
    model.with_structured_output.side_effect = structured
    with (
        open_sqlite_checkpointer(tmp_path / "checkpoints.sqlite") as checkpoint,
        open_memory_store(tmp_path / "memory.sqlite") as memory,
    ):
        runner = AgentRunner(
            WorkflowModels.from_single_model(model),
            checkpointer=checkpoint,
            memory_store=memory,
        )
        outcome = runner.run(
            RunRequest(
                repo_path=repo,
                objective="Explain the flag.",
                context_budget_tokens=16384,
                max_tool_calls_per_step=2,
                max_explorer_calls_per_step=3,
            )
        )
        assert outcome.termination_status == "accepted"
        assert prewarm_readiness(outcome, memory, repo)["ready"]
        assert calls == [
            "PlanProposal",
            "explorer",
            "explorer",
            "EvidenceOnlyCandidateResult",
            "SemanticAssessment",
        ]
        evaluation = next(
            event
            for event in outcome.trace.events
            if event.node == "evaluate_candidate"
        )
        assert evaluation.payload["evaluation_report"]["plan_coverage"][
            "approved_step_ids"
        ] == ["summarize"]
        count = len(calls)
        assert (
            runner.resume(ResumeRequest(thread_id=outcome.thread_id)).termination_status
            == "accepted"
        )
        assert len(calls) == count


def test_source_observation_schema_exposes_only_null_symbol():
    schema = EvidenceOnlyCandidateResult.model_json_schema()
    assert (
        schema["$defs"]["SourceObservation"]["properties"]["symbol"]["type"] == "null"
    )
