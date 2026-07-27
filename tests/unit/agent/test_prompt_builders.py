"""Contract tests for the dynamic Agent prompt harness."""

from langchain_core.messages import HumanMessage, SystemMessage

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.evaluation import (
    EvidenceExcerpt,
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


def test_agent_messages_include_ephemeral_task_and_plan_context() -> None:
    original_message = HumanMessage(
        content="Find and explain the authentication entry point."
    )
    plan = InvestigationPlan(
        task_summary="Trace authentication from entry point to implementation.",
        success_criteria=["Identify the entry point with file evidence."],
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
        active_step_id="locate-entry",
    )
    state: AgentNodeInput = {
        "messages": [original_message],
        "objective": "Find and explain the authentication entry point.",
        "active_objective": "Locate the authentication entry point.",
        "investigation_plan": plan,
        "step_tool_call_count": 2,
        "max_tool_calls_per_step": 7,
        "step_explorer_call_count": 1,
        "max_explorer_calls_per_step": 3,
    }

    messages = build_agent_messages(state)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    system_content = str(messages[0].content)
    assert system_content.startswith(INVESTIGATION_SYSTEM_PROMPT)
    assert state["objective"] in system_content
    assert state["active_objective"] in system_content
    assert plan.task_summary in system_content
    assert plan.success_criteria[0] in system_content
    assert plan.active_step_id in system_content
    assert plan.steps[0].objective in system_content
    assert plan.steps[1].objective in system_content
    assert "Remaining tool calls for the active step:\n5" in system_content
    assert "Remaining Explorer invocations for the active step:\n2" in system_content
    assert messages[1] is original_message
    assert state["messages"] == [original_message]


def test_agent_messages_describe_missing_plan_without_persisting_context() -> None:
    original_message = HumanMessage(content="Inspect the repository.")
    state: AgentNodeInput = {
        "messages": [original_message],
        "objective": "Inspect the repository.",
        "active_objective": "Inspect the repository structure.",
        "investigation_plan": None,
        "step_tool_call_count": 0,
        "max_tool_calls_per_step": 3,
        "step_explorer_call_count": 0,
        "max_explorer_calls_per_step": 2,
    }

    messages = build_agent_messages(state)

    system_content = str(messages[0].content)
    assert "No investigation plan is available." in system_content
    assert messages[1:] == [original_message]
    assert len(state["messages"]) == 1


def test_candidate_messages_keep_evidence_history_below_candidate_prompt() -> None:
    original_message = HumanMessage(content="Inspect authentication.")
    state: CandidateBuilderInput = {
        "messages": [original_message],
        "objective": "Explain authentication.",
        "active_objective": "Locate the authentication entry point.",
        "investigation_plan": None,
        "stop_reason": "step_explorer_budget_exhausted",
    }

    messages = build_candidate_messages(state)

    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == CANDIDATE_BUILDING_SYSTEM_PROMPT
    assert messages[1] is original_message
    assert isinstance(messages[-1], HumanMessage)
    request = str(messages[-1].content)
    assert state["objective"] in request
    assert state["active_objective"] in request
    assert "No investigation plan is available." in request
    assert state["stop_reason"] in request
    assert state["messages"] == [original_message]


def test_evaluator_messages_contain_only_isolated_evaluation_data() -> None:
    candidate = CandidateResult(
        relevant_files=["auth.py"],
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
    assert candidate.summary in request
    assert "example_issue" in request
    assert "1 | def authenticate():" in request
