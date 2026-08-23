"""Active Context integration across every model node and trace boundary."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.context import ContextPack
from repocompass.agent.evaluation import EvaluationReport, SemanticAssessment
from repocompass.agent.nodes import (
    AgentNodeInput,
    CandidateBuilderInput,
    EvaluatorInput,
    PlannerInput,
    ReplannerInput,
    create_agent_node,
    create_build_candidate_node,
    create_evaluate_candidate_node,
    create_plan_investigation_node,
    create_revise_plan_node,
)
from repocompass.agent.planning import InvestigationPlan, PlanStep
from repocompass.agent.runner import _normalize_update

CONTEXT_BUDGET = 16_384


def _plan() -> InvestigationPlan:
    return InvestigationPlan(
        steps=[
            PlanStep(
                id="locate-auth",
                objective="Locate authentication.",
                expected_evidence=["A source file and exact line range."],
                status="active",
            )
        ]
    )


def _candidate() -> CandidateResult:
    return CandidateResult(
        evidence=[
            EvidenceItem(
                path="auth.py",
                line_start=1,
                line_end=2,
                symbol="authenticate",
                claim="auth.py defines authenticate.",
            )
        ],
        summary="Authentication is defined in auth.py.",
        unresolved_questions=[],
    )


def _assessment() -> SemanticAssessment:
    return SemanticAssessment(
        active_step_completed=True,
        answers_objective=True,
        evidence_supports_summary=True,
        has_critical_omissions=False,
        missing_evidence=[],
        recommended_focus=[],
        summary="The candidate is supported.",
    )


def _report() -> EvaluationReport:
    return EvaluationReport(
        deterministic_issues=[],
        semantic_assessment=_assessment(),
    )


def _structured_model(response: object) -> MagicMock:
    model = MagicMock(spec=BaseChatModel)
    model.with_structured_output.return_value = RunnableLambda(lambda _: response)
    return model


def test_planner_node_publishes_active_context_pack() -> None:
    state: PlannerInput = {
        "objective": "Find authentication.",
        "max_tool_calls_per_step": 6,
        "max_explorer_calls_per_step": 3,
        "context_budget_tokens": CONTEXT_BUDGET,
        "context_build_count": 0,
    }

    update = create_plan_investigation_node(_structured_model(_plan()))(state)

    assert update["investigation_plan"] == _plan()
    assert isinstance(update["active_context_pack"], ContextPack)
    assert update["context_build_count"] == 1
    assert update["active_context_pack"].estimated_tokens <= CONTEXT_BUDGET


def test_planner_accepts_provider_include_raw_envelope() -> None:
    state: PlannerInput = {
        "objective": "Find authentication.",
        "max_tool_calls_per_step": 6,
        "max_explorer_calls_per_step": 3,
        "context_budget_tokens": CONTEXT_BUDGET,
        "context_build_count": 0,
    }
    response = {
        "raw": AIMessage(content="provider structured response"),
        "parsed": _plan(),
        "parsing_error": None,
    }

    update = create_plan_investigation_node(_structured_model(response))(state)

    assert update["investigation_plan"] == _plan()
    assert update["context_build_count"] == 1


def test_explorer_node_uses_selected_context_and_does_not_replay_history(
    tmp_path: Path,
) -> None:
    response = AIMessage(content="Continue with a bounded search.")
    received: list[list[object]] = []

    def capture(messages: list[object]) -> AIMessage:
        received.append(list(messages))
        return response

    model = MagicMock(spec=BaseChatModel)
    model.bind_tools.return_value = RunnableLambda(capture)
    state: AgentNodeInput = {
        "messages": [
            HumanMessage(content="stale private history"),
            AIMessage(content="hidden explorer narrative"),
            ToolMessage(
                content="src/auth.py:1:def authenticate():",
                tool_call_id="call-search",
                name="search_text",
                status="success",
            ),
        ],
        "repo_path": tmp_path,
        "objective": "Find authentication.",
        "investigation_plan": _plan(),
        "step_tool_call_count": 1,
        "max_tool_calls_per_step": 6,
        "step_explorer_call_count": 0,
        "max_explorer_calls_per_step": 3,
        "evaluation_report": None,
        "context_budget_tokens": CONTEXT_BUDGET,
        "context_build_count": 4,
    }

    update = create_agent_node(model)(state)
    rendered = str(received[0][1].content)

    assert "src/auth.py" in rendered
    assert "stale private history" not in rendered
    assert "hidden explorer narrative" not in rendered
    assert update["messages"] == [response]
    assert isinstance(update["active_context_pack"], ContextPack)
    assert update["context_build_count"] == 5


def test_candidate_node_publishes_the_exact_pack_used_for_invocation(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    received: list[list[object]] = []
    model = MagicMock(spec=BaseChatModel)
    model.with_structured_output.return_value = RunnableLambda(
        lambda messages: received.append(list(messages)) or candidate
    )
    state: CandidateBuilderInput = {
        "messages": [
            ToolMessage(
                content=(
                    "File: auth.py\nLines: 1-2\n"
                    "1 | def authenticate():\n2 |     return True"
                ),
                tool_call_id="call-read",
                name="read_code",
                status="success",
            )
        ],
        "repo_path": tmp_path,
        "objective": "Find authentication.",
        "investigation_plan": _plan(),
        "step_stop_reason": "evidence_sufficient",
        "context_budget_tokens": CONTEXT_BUDGET,
        "context_build_count": 2,
    }

    update = create_build_candidate_node(model)(state)

    assert update["candidate_result"] == candidate
    assert isinstance(update["active_context_pack"], ContextPack)
    assert update["context_build_count"] == 3
    assert "def authenticate" in str(received[0][1].content)


def test_evaluator_node_publishes_isolated_context_pack(tmp_path: Path) -> None:
    (tmp_path / "auth.py").write_text(
        "def authenticate():\n    return True\n",
        encoding="utf-8",
    )
    state: EvaluatorInput = {
        "repo_path": tmp_path,
        "objective": "Find authentication.",
        "investigation_plan": _plan(),
        "candidate_result": _candidate(),
        "context_budget_tokens": CONTEXT_BUDGET,
        "context_build_count": 7,
    }

    update = create_evaluate_candidate_node(_structured_model(_assessment()))(state)

    assert update["evaluation_report"] == _report()
    assert isinstance(update["active_context_pack"], ContextPack)
    assert update["context_build_count"] == 8


def test_replanner_node_publishes_required_context_pack() -> None:
    state: ReplannerInput = {
        "objective": "Find authentication.",
        "investigation_plan": _plan(),
        "candidate_result": _candidate(),
        "evaluation_report": EvaluationReport(
            deterministic_issues=[],
            semantic_assessment=SemanticAssessment(
                active_step_completed=False,
                answers_objective=False,
                evidence_supports_summary=True,
                has_critical_omissions=True,
                missing_evidence=["Caller evidence is missing."],
                recommended_focus=["Search for callers."],
                summary="More evidence is required.",
            ),
        ),
        "human_feedback": "Inspect middleware too.",
        "max_tool_calls_per_step": 6,
        "max_explorer_calls_per_step": 3,
        "plan_revision_count": 0,
        "max_plan_revisions": 2,
        "context_budget_tokens": CONTEXT_BUDGET,
        "context_build_count": 9,
    }

    update = create_revise_plan_node(_structured_model(_plan()))(state)

    assert update["investigation_plan"] == _plan()
    assert isinstance(update["active_context_pack"], ContextPack)
    assert update["context_build_count"] == 10


def test_trace_normalization_exposes_context_decisions_without_item_content() -> None:
    planner_state: PlannerInput = {
        "objective": "highly-sensitive-objective-text",
        "max_tool_calls_per_step": 6,
        "max_explorer_calls_per_step": 3,
        "context_budget_tokens": CONTEXT_BUDGET,
        "context_build_count": 0,
    }
    pack = create_plan_investigation_node(_structured_model(_plan()))(planner_state)[
        "active_context_pack"
    ]

    normalized = _normalize_update({"active_context_pack": pack})
    encoded = json.dumps(normalized, sort_keys=True)

    assert normalized["active_context_pack"]["pack_id"] == pack.pack_id
    assert normalized["active_context_pack"]["selected_count"] == len(pack.items)
    assert "required_objective" in encoded
    assert "highly-sensitive-objective-text" not in encoded
