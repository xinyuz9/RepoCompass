"""Contract tests for the dynamic Agent prompt harness."""

from langchain_core.messages import HumanMessage, SystemMessage

from repocompass.agent.contracts import InvestigationPlan, PlanStep
from repocompass.agent.prompts import (
    INVESTIGATION_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    build_agent_messages,
    build_planner_messages,
)
from repocompass.agent.state import AgentNodeInput, PlannerInput


def test_planner_messages_are_isolated_from_repository_content() -> None:
    state: PlannerInput = {
        "objective": "Find and explain the authentication entry point.",
        "max_tool_calls": 7,
        "max_iterations": 3,
    }

    messages = build_planner_messages(state)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == PLANNING_SYSTEM_PROMPT
    assert isinstance(messages[1], HumanMessage)
    request = str(messages[1].content)
    assert state["objective"] in request
    assert "Available tool calls: 7" in request
    assert "Available Explorer invocations: 3" in request
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
        "tool_call_count": 2,
        "max_tool_calls": 7,
        "iteration_count": 1,
        "max_iterations": 3,
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
    assert "Remaining tool calls:\n5" in system_content
    assert "Remaining Explorer invocations:\n2" in system_content
    assert messages[1] is original_message
    assert state["messages"] == [original_message]


def test_agent_messages_describe_missing_plan_without_persisting_context() -> None:
    original_message = HumanMessage(content="Inspect the repository.")
    state: AgentNodeInput = {
        "messages": [original_message],
        "objective": "Inspect the repository.",
        "active_objective": "Inspect the repository structure.",
        "investigation_plan": None,
        "tool_call_count": 0,
        "max_tool_calls": 3,
        "iteration_count": 0,
        "max_iterations": 2,
    }

    messages = build_agent_messages(state)

    system_content = str(messages[0].content)
    assert "No investigation plan is available." in system_content
    assert messages[1:] == [original_message]
    assert len(state["messages"]) == 1
