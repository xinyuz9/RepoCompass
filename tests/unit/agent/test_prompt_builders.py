"""Contract tests for the dynamic Agent prompt harness."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.evaluation import (
    EvaluationReport,
    SemanticAssessment,
)
from repocompass.agent.nodes import (
    AgentNodeInput,
    CandidateBuilderInput,
    EvaluatorInput,
    PlannerInput,
    create_agent_node,
    create_build_candidate_node,
    create_evaluate_candidate_node,
    create_plan_investigation_node,
)
from repocompass.agent.planning import InvestigationPlan, PlanStep
from repocompass.agent.prompts import (
    CANDIDATE_EVIDENCE_ONLY_SYSTEM_PROMPT,
    EVALUATION_SYSTEM_PROMPT,
    INVESTIGATION_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT,
)


def _selected_content(message: HumanMessage) -> str:
    """Decode the canonical JSONL records from a rendered Active Context."""

    rendered = str(message.content)
    _, separator, jsonl = rendered.partition("\n\n")
    assert separator
    records = [json.loads(line) for line in jsonl.splitlines() if line]
    assert all(isinstance(record.get("content"), str) for record in records)
    return "\n".join(record["content"] for record in records)


def _capture(factory, state, response=None):
    """Inspect the actual node invocation, not a separate prompt-only path."""
    received = []
    model = MagicMock(spec=BaseChatModel)
    invocation = RunnableLambda(
        lambda messages: (
            received.append(messages) or response or AIMessage(content="Stop.")
        )
    )
    model.bind_tools.return_value = invocation
    model.with_structured_output.return_value = invocation
    factory(model)(state)
    assert len(received) == 1
    return received[0]


def test_planner_messages_are_isolated_from_repository_content() -> None:
    state: PlannerInput = {
        "objective": "Find and explain the authentication entry point.",
        "max_tool_calls_per_step": 7,
        "max_explorer_calls_per_step": 3,
    }

    plan = InvestigationPlan(
        steps=[
            PlanStep(
                id="inspect",
                objective="Inspect authentication.",
                expected_evidence=["Source lines."],
                status="active",
            )
        ]
    )
    messages = _capture(create_plan_investigation_node, state, plan)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == PLANNING_SYSTEM_PROMPT
    assert isinstance(messages[1], HumanMessage)
    request = _selected_content(messages[1])
    assert state["objective"] in request
    assert "Available tool calls per plan step: 7" in request
    assert "Available Explorer invocations per plan step: 3" in request
    assert "repo_path" not in request


def test_agent_messages_expose_only_the_active_plan_step(
    tmp_path: Path,
) -> None:
    original_message = HumanMessage(
        content="Find and explain the authentication entry point."
    )
    plan = InvestigationPlan(
        steps=[
            PlanStep(
                id="locate-entry",
                objective="Locate the authentication entry point.",
                expected_evidence=["A repository-relative file and line range."],
                status="active",
            ),
            PlanStep(
                id="trace-flow",
                objective="Trace the authentication call flow.",
                expected_evidence=["Verified caller and callee locations."],
                status="pending",
            ),
        ],
    )
    state: AgentNodeInput = {
        "messages": [original_message],
        "repo_path": tmp_path,
        "objective": "Find and explain the authentication entry point.",
        "investigation_plan": plan,
        "step_tool_call_count": 2,
        "max_tool_calls_per_step": 7,
        "step_explorer_call_count": 1,
        "max_explorer_calls_per_step": 3,
        "evaluation_report": None,
        "context_budget_tokens": 16_384,
        "context_build_count": 0,
    }

    messages = _capture(create_agent_node, state)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == INVESTIGATION_SYSTEM_PROMPT
    assert isinstance(messages[1], HumanMessage)
    selected_content = _selected_content(messages[1])
    assert state["objective"] in selected_content
    assert plan.active_step.objective in selected_content
    assert plan.active_step.id in selected_content
    assert plan.active_step.expected_evidence[0] in selected_content
    assert plan.steps[1].objective not in selected_content
    assert plan.steps[1].expected_evidence[0] not in selected_content
    assert "Do not\ninvestigate, anticipate, or complete pending plan steps." in str(
        messages[0].content
    )
    assert "Remaining tool calls for the active step:\n5" in selected_content
    assert "Remaining Explorer invocations for the active step:\n2" in selected_content
    assert messages[1] is not original_message
    assert state["messages"] == [original_message]


def test_agent_messages_require_an_investigation_plan() -> None:
    original_message = HumanMessage(content="Inspect the repository.")
    state: AgentNodeInput = {
        "messages": [original_message],
        "repo_path": Path("."),
        "objective": "Inspect the repository.",
        "investigation_plan": None,
        "step_tool_call_count": 0,
        "max_tool_calls_per_step": 3,
        "step_explorer_call_count": 0,
        "max_explorer_calls_per_step": 2,
        "evaluation_report": None,
        "context_budget_tokens": 16_384,
        "context_build_count": 0,
    }

    with pytest.raises(ValueError, match="InvestigationPlan"):
        _capture(create_agent_node, state)


def test_agent_messages_select_ephemeral_evaluator_feedback(
    tmp_path: Path,
) -> None:
    original_message = HumanMessage(content="Inspect the repository.")
    assessment = SemanticAssessment(
        active_step_completed=False,
        answers_objective=False,
        evidence_supports_summary=False,
        has_critical_omissions=True,
        missing_evidence=["The implementation location is missing."],
        recommended_focus=["Search for the authentication entry point."],
        summary="The current candidate is incomplete.",
    )
    report = EvaluationReport(
        deterministic_issues=[],
        semantic_assessment=assessment,
    )
    plan = InvestigationPlan(
        steps=[
            PlanStep(
                id="locate-authentication",
                objective="Locate authentication.",
                expected_evidence=["The implementation location."],
                status="active",
            ),
            PlanStep(
                id="trace-flow",
                objective="Trace callers after locating the entry point.",
                expected_evidence=["Verified caller and callee locations."],
                status="pending",
            ),
        ]
    )
    state: AgentNodeInput = {
        "messages": [original_message],
        "repo_path": tmp_path,
        "objective": "Explain authentication.",
        "investigation_plan": plan,
        "step_tool_call_count": 1,
        "max_tool_calls_per_step": 3,
        "step_explorer_call_count": 1,
        "max_explorer_calls_per_step": 2,
        "evaluation_report": report,
        "context_budget_tokens": 16_384,
        "context_build_count": 0,
    }

    messages = _capture(create_agent_node, state)

    assert len(messages) == 2
    assert messages[1] is not original_message
    assert isinstance(messages[1], HumanMessage)
    feedback = _selected_content(messages[1])
    assert assessment.missing_evidence[0] in feedback
    assert assessment.recommended_focus[0] in feedback
    assert state["messages"] == [original_message]


def test_candidate_messages_render_tool_results_without_tool_protocol_history(
    tmp_path: Path,
) -> None:
    original_message = HumanMessage(content="Inspect authentication.")
    tool_request = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "read_code",
                "args": {
                    "relative_path": "auth.py",
                    "start_line": 1,
                    "end_line": 3,
                },
                "id": "call-read",
                "type": "tool_call",
            }
        ],
    )
    tool_result = ToolMessage(
        content="File: auth.py\nLines: 1-3\n1 | def authenticate():",
        tool_call_id="call-read",
        name="read_code",
    )
    plan = InvestigationPlan(
        steps=[
            PlanStep(
                id="locate-entry",
                objective="Locate the authentication entry point.",
                expected_evidence=["An evidence-backed source location."],
                status="active",
            ),
            PlanStep(
                id="trace-flow",
                objective="Trace callers after locating the entry point.",
                expected_evidence=["Verified caller and callee locations."],
                status="pending",
            ),
        ]
    )
    state: CandidateBuilderInput = {
        "messages": [original_message, tool_request, tool_result],
        "repo_path": tmp_path,
        "objective": "Explain authentication.",
        "investigation_plan": plan,
        "step_stop_reason": "step_explorer_budget_exhausted",
        "context_budget_tokens": 16_384,
        "context_build_count": 0,
    }

    messages = _capture(
        create_build_candidate_node,
        state,
        CandidateResult(
            evidence=[],
            summary="No verified answer.",
            unresolved_questions=["Read source."],
        ),
    )

    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == CANDIDATE_EVIDENCE_ONLY_SYSTEM_PROMPT
    assert len(messages) == 2
    assert isinstance(messages[1], HumanMessage)
    request = _selected_content(messages[1])
    assert state["objective"] in request
    assert plan.active_step.objective in request
    assert plan.active_step.expected_evidence[0] in request
    assert plan.steps[1].objective not in request
    assert plan.steps[1].expected_evidence[0] not in request
    assert state["step_stop_reason"] in request
    assert '"tool": "read_code"' in request
    assert "1 | def authenticate():" in request
    assert all(not isinstance(message, ToolMessage) for message in messages)
    assert all(not getattr(message, "tool_calls", []) for message in messages)
    assert state["messages"] == [original_message, tool_request, tool_result]


def test_evaluator_messages_contain_only_isolated_evaluation_data(
    tmp_path: Path,
) -> None:
    (tmp_path / "auth.py").write_text("def authenticate():\n    return True\n")
    candidate = CandidateResult(
        evidence=[
            EvidenceItem(
                path="auth.py", line_start=1, claim="auth.py defines authentication."
            ),
            EvidenceItem(path="missing.py", line_start=1, claim="Unverified claim."),
        ],
        summary="Authentication is defined in auth.py.",
        unresolved_questions=[],
    )
    step = PlanStep(
        id="locate-authentication",
        objective="Locate authentication.",
        expected_evidence=["A source location for authentication."],
        status="active",
    )
    state: EvaluatorInput = {
        "repo_path": tmp_path,
        "objective": "Find authentication.",
        "investigation_plan": InvestigationPlan(steps=[step]),
        "candidate_result": candidate,
        "context_budget_tokens": 16_384,
        "context_build_count": 0,
    }
    assessment = SemanticAssessment(
        active_step_completed=False,
        answers_objective=False,
        evidence_supports_summary=False,
        has_critical_omissions=True,
        missing_evidence=["Read the missing source."],
        recommended_focus=[],
        summary="The candidate needs repair.",
    )
    messages = _capture(create_evaluate_candidate_node, state, assessment)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == EVALUATION_SYSTEM_PROMPT
    assert isinstance(messages[1], HumanMessage)
    request = _selected_content(messages[1])
    assert state["objective"] in request
    assert step.objective in request
    assert step.expected_evidence[0] in request
    assert candidate.summary in request
    assert "path_not_found" in request
    assert "1 | def authenticate():" in request
