"""Contract tests for the dynamic Agent prompt harness."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.evaluation import (
    EvaluationReport,
    EvidenceExcerpt,
    SemanticAssessment,
    ValidationIssue,
)
from repocompass.agent.nodes import (
    AgentNodeInput,
    CandidateBuilderInput,
    EvaluationPromptInput,
    PlannerInput,
    build_agent_messages,
    build_candidate_messages,
    build_evaluator_messages,
    build_planner_messages,
)
from repocompass.agent.planning import InvestigationPlan, PlanStep
from repocompass.agent.prompts import (
    CANDIDATE_BUILDING_SYSTEM_PROMPT,
    EVALUATION_SYSTEM_PROMPT,
    INVESTIGATION_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT,
)


def test_planner_messages_are_isolated_from_repository_content() -> None:
    state: PlannerInput = {
        "objective": "Find and explain the authentication entry point.",
        "max_tool_calls_per_step": 7,
        "max_explorer_calls_per_step": 3,
    }

    messages = build_planner_messages(state)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == PLANNING_SYSTEM_PROMPT
    assert isinstance(messages[1], HumanMessage)
    request = str(messages[1].content)
    assert state["objective"] in request
    assert "Available tool calls per plan step: 7" in request
    assert "Available Explorer invocations per plan step: 3" in request
    assert "repo_path" not in request


def test_agent_messages_expose_only_the_active_plan_step() -> None:
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
        "objective": "Find and explain the authentication entry point.",
        "investigation_plan": plan,
        "step_tool_call_count": 2,
        "max_tool_calls_per_step": 7,
        "step_explorer_call_count": 1,
        "max_explorer_calls_per_step": 3,
        "evaluation_report": None,
    }

    messages = build_agent_messages(state)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    system_content = str(messages[0].content)
    assert system_content.startswith(INVESTIGATION_SYSTEM_PROMPT)
    assert state["objective"] in system_content
    assert plan.active_step.objective in system_content
    assert plan.active_step.id in system_content
    assert plan.active_step.expected_evidence[0] in system_content
    assert plan.steps[1].objective not in system_content
    assert plan.steps[1].expected_evidence[0] not in system_content
    assert "Do not\ninvestigate, anticipate, or complete pending plan steps." in (
        system_content
    )
    assert "Remaining tool calls for the active step:\n5" in system_content
    assert "Remaining Explorer invocations for the active step:\n2" in system_content
    assert messages[1] is original_message
    assert state["messages"] == [original_message]


def test_agent_messages_require_an_investigation_plan() -> None:
    original_message = HumanMessage(content="Inspect the repository.")
    state: AgentNodeInput = {
        "messages": [original_message],
        "objective": "Inspect the repository.",
        "investigation_plan": None,
        "step_tool_call_count": 0,
        "max_tool_calls_per_step": 3,
        "step_explorer_call_count": 0,
        "max_explorer_calls_per_step": 2,
        "evaluation_report": None,
    }

    with pytest.raises(ValueError, match="InvestigationPlan"):
        build_agent_messages(state)


def test_agent_messages_append_ephemeral_evaluator_feedback() -> None:
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
        "objective": "Explain authentication.",
        "investigation_plan": plan,
        "step_tool_call_count": 1,
        "max_tool_calls_per_step": 3,
        "step_explorer_call_count": 1,
        "max_explorer_calls_per_step": 2,
        "evaluation_report": report,
    }

    messages = build_agent_messages(state)

    assert messages[1] is original_message
    assert isinstance(messages[-1], HumanMessage)
    feedback = str(messages[-1].content)
    assert assessment.missing_evidence[0] in feedback
    assert assessment.recommended_focus[0] in feedback
    assert state["messages"] == [original_message]


def test_candidate_messages_render_tool_results_without_tool_protocol_history() -> None:
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
        "objective": "Explain authentication.",
        "investigation_plan": plan,
        "step_stop_reason": "step_explorer_budget_exhausted",
    }

    messages = build_candidate_messages(state)

    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == CANDIDATE_BUILDING_SYSTEM_PROMPT
    assert len(messages) == 2
    assert isinstance(messages[1], HumanMessage)
    request = str(messages[1].content)
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


def test_evaluator_messages_contain_only_isolated_evaluation_data() -> None:
    candidate = CandidateResult(
        evidence=[
            EvidenceItem(
                path="auth.py",
                line_start=1,
                claim="auth.py defines authentication.",
            )
        ],
        summary="Authentication is defined in auth.py.",
        unresolved_questions=[],
    )
    state: EvaluationPromptInput = {
        "objective": "Find authentication.",
        "active_step": PlanStep(
            id="locate-authentication",
            objective="Locate authentication.",
            expected_evidence=["A source location for authentication."],
            status="active",
        ),
        "candidate_result": candidate,
        "deterministic_issues": (
            ValidationIssue(
                code="example_issue",
                message="Example deterministic finding.",
                path="auth.py",
                evidence_index=0,
            ),
        ),
        "evidence_pack": (
            EvidenceExcerpt(
                evidence_index=0,
                path="auth.py",
                line_start=1,
                line_end=1,
                symbol=None,
                claim="auth.py defines authentication.",
                content="1 | def authenticate():",
            ),
        ),
    }

    messages = build_evaluator_messages(state)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == EVALUATION_SYSTEM_PROMPT
    assert isinstance(messages[1], HumanMessage)
    request = str(messages[1].content)
    assert state["objective"] in request
    assert state["active_step"].objective in request
    assert state["active_step"].expected_evidence[0] in request
    assert candidate.summary in request
    assert "example_issue" in request
    assert "1 | def authenticate():" in request
