"""Unit tests for the compiled RepoCompass agent graph."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    InvalidToolCall,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableLambda

import repocompass.tools.gateway as gateway_module
from repocompass.agent.candidate_output import EvidenceOnlyCandidateResult
from repocompass.agent.evaluation import SemanticAssessment
from repocompass.agent.graph import create_agent_graph
from repocompass.agent.planning import InvestigationPlan, PlanStep
from repocompass.agent.result import FinalResult
from repocompass.agent.state import create_initial_state
from repocompass.tools.contracts import (
    NoProgressReason,
    ToolDecisionReason,
    decode_no_progress_decision,
    decode_tool_audit_record,
)
from tests.model_doubles import (
    assert_final_result_core,
    candidate_from_final_result,
    with_scripted_planner,
)


def _create_scripted_model(
    agent_responses: list[AIMessage],
    final_result: FinalResult,
) -> MagicMock:
    """Create a model double with deterministic Explorer and candidate responses."""

    response_iterator = iter(agent_responses)

    def next_agent_response(_: object) -> AIMessage:
        try:
            return next(response_iterator)
        except StopIteration:
            return AIMessage(content="No additional repository evidence is available.")

    model = MagicMock(spec=BaseChatModel)
    model.bind_tools.return_value = RunnableLambda(next_agent_response)
    model.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(final_result)
    )
    return model


def _semantic_assessment(
    *,
    active_step_completed: bool,
    answers_objective: bool,
) -> SemanticAssessment:
    return SemanticAssessment(
        active_step_completed=active_step_completed,
        answers_objective=answers_objective,
        evidence_supports_summary=answers_objective,
        has_critical_omissions=not answers_objective,
        missing_evidence=[] if answers_objective else ["More evidence is required."],
        recommended_focus=[] if answers_objective else ["Continue repository search."],
        summary="Accepted." if answers_objective else "More investigation is required.",
    )


def test_graph_finalizes_when_agent_returns_no_tool_calls(tmp_path: Path) -> None:
    """A tool-free agent response should be finalized and terminate normally."""

    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "example.py").write_text("value = True\n", encoding="utf-8")
    agent_response = AIMessage(content="Enough evidence is already available.")
    expected_result = FinalResult(
        relevant_files=["src/example.py"],
        evidence=["src/example.py contains the relevant implementation."],
        summary="The relevant implementation was identified.",
        unresolved_questions=[],
    )

    model = _create_scripted_model([agent_response], expected_result)

    user_message = HumanMessage(content="Find the relevant implementation.")
    initial_state = create_initial_state(
        tmp_path,
        str(user_message.content),
        max_tool_calls_per_step=3,
        max_explorer_calls_per_step=3,
    )

    final_state = create_agent_graph(with_scripted_planner(model)).invoke(initial_state)

    assert_final_result_core(final_state["final_result"], expected_result)
    assert final_state["candidate_result"] == candidate_from_final_result(
        expected_result
    )
    assert final_state["step_stop_reason"] is None
    assert final_state["termination_status"] == "accepted"
    assert final_state["step_tool_call_count"] == 0
    assert final_state["step_explorer_call_count"] == 1
    messages = final_state["messages"]
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == user_message.content
    assert messages[1] == agent_response
    model.bind_tools.assert_called_once()
    model.with_structured_output.assert_called_once_with(
        EvidenceOnlyCandidateResult,
        method="function_calling",
        include_raw=True,
    )


def test_graph_finalizes_when_agent_returns_tool_calls(tmp_path: Path) -> None:
    """A valid tool call should execute, return to Agent, and then finalize."""

    auth_file = tmp_path / "auth.py"
    auth_file.write_text(
        "def authenticate():\n    return True\n",
        encoding="utf-8",
    )

    tree_call = ToolCall(
        name="list_tree",
        args={
            "max_depth": 2,
            "max_entries": 10,
        },
        id="call-tree",
        type="tool_call",
    )

    tool_request = AIMessage(
        content="",
        tool_calls=[tree_call],
    )

    enough_evidence_response = AIMessage(
        content="Enough evidence is already available."
    )

    expected_result = FinalResult(
        relevant_files=["auth.py"],
        evidence=["auth.py defines authenticate."],
        summary="The relevant implementation was identified.",
        unresolved_questions=[],
    )

    model = _create_scripted_model(
        [tool_request, enough_evidence_response],
        expected_result,
    )

    user_message = HumanMessage(content="Find the relevant implementation.")
    initial_state = create_initial_state(
        tmp_path,
        str(user_message.content),
        max_tool_calls_per_step=3,
        max_explorer_calls_per_step=3,
    )

    final_state = create_agent_graph(with_scripted_planner(model)).invoke(initial_state)
    messages = final_state["messages"]

    assert_final_result_core(final_state["final_result"], expected_result)
    assert final_state["evaluation_report"].passed is True
    assert final_state["step_stop_reason"] is None
    assert final_state["step_tool_call_count"] == 1
    assert final_state["step_explorer_call_count"] == 2
    assert len(messages) == 4
    assert messages[1] == tool_request
    assert isinstance(messages[2], ToolMessage)
    assert messages[3] == enough_evidence_response
    assert messages[2].tool_call_id == "call-tree"
    assert "auth.py" in str(messages[2].content)
    assert "repo_path" not in tree_call["args"]
    audit_payloads = final_state["tool_audit_records"]
    assert len(audit_payloads) == 1
    audit = decode_tool_audit_record(audit_payloads[0])
    assert audit.decision.reason is ToolDecisionReason.ALLOWED
    model.bind_tools.assert_called_once()
    model.with_structured_output.assert_called_once_with(
        EvidenceOnlyCandidateResult,
        method="function_calling",
        include_raw=True,
    )


def test_graph_rejects_batch_exceeding_budget_then_exhausts_revisions(
    tmp_path: Path,
) -> None:
    """An oversized batch is rejected before bounded retries eventually exhaust."""

    tree_call = ToolCall(
        name="list_tree",
        args={"max_depth": 2, "max_entries": 10},
        id="call-tree",
        type="tool_call",
    )
    search_call = ToolCall(
        name="search_text",
        args={"query": "authenticate", "max_results": 10},
        id="call-search",
        type="tool_call",
    )
    oversized_request = AIMessage(
        content="",
        tool_calls=[tree_call, search_call],
    )
    expected_result = FinalResult(
        relevant_files=[],
        evidence=[],
        summary="The investigation stopped before the requested batch could run.",
        unresolved_questions=["Which files contain the authentication logic?"],
    )
    model = _create_scripted_model([oversized_request], expected_result)
    user_message = HumanMessage(content="Find the authentication logic.")
    initial_state = create_initial_state(
        tmp_path,
        str(user_message.content),
        max_tool_calls_per_step=1,
        max_explorer_calls_per_step=3,
    )

    final_state = create_agent_graph(with_scripted_planner(model)).invoke(initial_state)
    messages = final_state["messages"]

    assert_final_result_core(final_state["final_result"], expected_result)
    assert final_state["step_stop_reason"] is None
    assert final_state["termination_status"] == "exhausted"
    assert final_state["plan_revision_count"] == 2
    assert final_state["next_action"] == "exhausted"
    assert final_state["step_tool_call_count"] == 0
    # Empty evidence no longer becomes completed solely on scripted model
    # approval: each revision remains active until its retry budget is spent.
    assert final_state["step_explorer_call_count"] == 3
    assert len(messages) == 10  # initial refusal + two bounded three-call revisions
    assert not any(
        step.status == "completed" for step in final_state["investigation_plan"].steps
    )
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == user_message.content
    assert messages[1] == oversized_request
    first_rejection = messages[2]
    second_rejection = messages[3]
    assert isinstance(first_rejection, ToolMessage)
    assert isinstance(second_rejection, ToolMessage)
    assert first_rejection.tool_call_id == "call-tree"
    assert second_rejection.tool_call_id == "call-search"
    assert first_rejection.status == "error"
    assert second_rejection.status == "error"
    assert json.loads(str(first_rejection.content)) == {
        "error": {"reason": "budget_exhausted"}
    }
    assert json.loads(str(second_rejection.content)) == {
        "error": {"reason": "budget_exhausted"}
    }
    assert [
        decode_tool_audit_record(payload).decision.reason
        for payload in final_state["tool_audit_records"]
    ] == [ToolDecisionReason.BUDGET_EXHAUSTED] * 2
    assert all(isinstance(message, AIMessage) for message in messages[4:])


def test_graph_handles_mixed_invalid_batch_without_executing_valid_calls(
    tmp_path: Path,
) -> None:
    """A mixed batch should reject every call and return to Agent for correction."""

    (tmp_path / "auth.py").write_text(
        "def authenticate():\n    return True\n",
        encoding="utf-8",
    )
    tree_call = ToolCall(
        name="list_tree",
        args={"max_depth": 2, "max_entries": 10},
        id="call-tree",
        type="tool_call",
    )
    invalid_search_call = InvalidToolCall(
        type="invalid_tool_call",
        id="call-invalid-search",
        name="search_text",
        args='{"query":',
        error="Invalid JSON arguments",
    )
    mixed_request = AIMessage(
        content="",
        tool_calls=[tree_call],
        invalid_tool_calls=[invalid_search_call],
    )
    corrected_response = AIMessage(
        content="Continue using only the evidence already available."
    )
    expected_result = FinalResult(
        relevant_files=["auth.py"],
        evidence=["auth.py contains the authentication implementation."],
        summary="Authentication is implemented in auth.py.",
        unresolved_questions=[],
    )
    model = _create_scripted_model(
        [mixed_request, corrected_response],
        expected_result,
    )
    user_message = HumanMessage(content="Find the authentication logic.")
    initial_state = create_initial_state(
        tmp_path,
        str(user_message.content),
        max_tool_calls_per_step=3,
        max_explorer_calls_per_step=3,
    )

    final_state = create_agent_graph(with_scripted_planner(model)).invoke(initial_state)
    messages = final_state["messages"]

    assert_final_result_core(final_state["final_result"], expected_result)
    assert final_state["step_stop_reason"] is None
    assert final_state["termination_status"] == "accepted"
    assert final_state["step_tool_call_count"] == 0
    assert final_state["step_explorer_call_count"] == 2
    assert len(messages) == 5
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == user_message.content
    assert messages[1] == mixed_request
    invalid_error = messages[2]
    withheld_valid_call = messages[3]
    assert isinstance(invalid_error, ToolMessage)
    assert isinstance(withheld_valid_call, ToolMessage)
    assert invalid_error.tool_call_id == "call-invalid-search"
    assert invalid_error.status == "error"
    assert json.loads(str(invalid_error.content)) == {
        "error": {"reason": "invalid_arguments"}
    }
    assert "Invalid JSON arguments" not in str(invalid_error.content)
    assert withheld_valid_call.tool_call_id == "call-tree"
    assert withheld_valid_call.status == "error"
    assert json.loads(str(withheld_valid_call.content)) == {
        "error": {"reason": "invalid_arguments"}
    }
    assert "auth.py" not in str(withheld_valid_call.content)
    assert [
        decode_tool_audit_record(payload).decision.reason
        for payload in final_state["tool_audit_records"]
    ] == [ToolDecisionReason.INVALID_ARGUMENTS] * 2
    assert messages[4] == corrected_response


def test_graph_returns_recoverable_tool_invocation_error_to_agent(
    tmp_path: Path,
) -> None:
    """A schema error should become a ToolMessage before Agent runs again."""

    (tmp_path / "auth.py").write_text(
        "def authenticate():\n    return True\n",
        encoding="utf-8",
    )
    invalid_schema_call = ToolCall(
        name="list_tree",
        args={"max_depth": "not-an-int", "max_entries": 10},
        id="call-invalid-schema",
        type="tool_call",
    )
    invalid_request = AIMessage(
        content="",
        tool_calls=[invalid_schema_call],
    )
    recovered_response = AIMessage(content="Answer using the available evidence.")
    expected_result = FinalResult(
        relevant_files=["auth.py"],
        evidence=["auth.py defines authenticate."],
        summary="The invalid request was recovered and authentication was located.",
        unresolved_questions=[],
    )
    model = _create_scripted_model(
        [invalid_request, recovered_response],
        expected_result,
    )
    user_message = HumanMessage(content="Inspect the repository.")
    initial_state = create_initial_state(
        tmp_path,
        str(user_message.content),
        max_tool_calls_per_step=3,
        max_explorer_calls_per_step=3,
    )

    final_state = create_agent_graph(with_scripted_planner(model)).invoke(initial_state)
    messages = final_state["messages"]

    assert_final_result_core(final_state["final_result"], expected_result)
    assert final_state["step_stop_reason"] is None
    assert final_state["termination_status"] == "accepted"
    assert final_state["step_tool_call_count"] == 0
    assert final_state["step_explorer_call_count"] == 2
    assert len(messages) == 4
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == user_message.content
    assert messages[1] == invalid_request
    tool_error = messages[2]
    assert isinstance(tool_error, ToolMessage)
    assert tool_error.name == "list_tree"
    assert tool_error.tool_call_id == "call-invalid-schema"
    assert tool_error.status == "error"
    assert json.loads(str(tool_error.content)) == {
        "error": {"reason": "invalid_arguments"}
    }
    assert "max_depth" not in str(tool_error.content)
    assert len(final_state["tool_audit_records"]) == 1
    audit = decode_tool_audit_record(final_state["tool_audit_records"][0])
    assert audit.decision.reason is ToolDecisionReason.INVALID_ARGUMENTS
    assert messages[3] == recovered_response


def test_graph_finalizes_without_extra_agent_invocation_at_iteration_limit(
    tmp_path: Path,
) -> None:
    """Tool evidence is retained, but Agent cannot run after its last iteration."""

    (tmp_path / "auth.py").write_text(
        "def authenticate():\n    return True\n",
        encoding="utf-8",
    )
    tree_call = ToolCall(
        name="list_tree",
        args={"max_depth": 2, "max_entries": 10},
        id="call-tree",
        type="tool_call",
    )
    tool_request = AIMessage(content="", tool_calls=[tree_call])
    expected_result = FinalResult(
        relevant_files=["auth.py"],
        evidence=["The list_tree result includes auth.py."],
        summary="The investigation stopped after collecting directory evidence.",
        unresolved_questions=["Where is authenticate called?"],
    )
    agent_inputs: list[list[object]] = []

    def one_agent_response(messages: list[object]) -> AIMessage:
        agent_inputs.append(list(messages))
        return tool_request

    model = MagicMock(spec=BaseChatModel)
    model.bind_tools.return_value = RunnableLambda(one_agent_response)
    model.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(expected_result)
    )
    user_message = HumanMessage(content="Find the authentication logic.")
    initial_state = create_initial_state(
        tmp_path,
        str(user_message.content),
        max_tool_calls_per_step=3,
        max_explorer_calls_per_step=1,
    )

    final_state = create_agent_graph(with_scripted_planner(model)).invoke(initial_state)

    assert len(agent_inputs) == 1
    assert final_state["step_explorer_call_count"] == 1
    assert final_state["step_tool_call_count"] == 1
    assert final_state["step_stop_reason"] == "step_explorer_budget_exhausted"
    assert final_state["termination_status"] == "accepted"
    assert_final_result_core(final_state["final_result"], expected_result)
    assert isinstance(final_state["messages"][-1], ToolMessage)


def test_graph_continues_the_active_step_after_incomplete_evaluation(
    tmp_path: Path,
) -> None:
    (tmp_path / "auth.py").write_text("value = True\n", encoding="utf-8")
    expected_result = FinalResult(
        relevant_files=["auth.py"],
        evidence=["auth.py contains the relevant implementation."],
        summary="The relevant implementation is in auth.py.",
        unresolved_questions=[],
    )
    model = _create_scripted_model(
        [
            AIMessage(content="Initial evidence collected."),
            AIMessage(content="The missing evidence has now been addressed."),
        ],
        expected_result,
    )
    models = with_scripted_planner(
        model,
        semantic_assessments=[
            _semantic_assessment(
                active_step_completed=False,
                answers_objective=False,
            ),
            _semantic_assessment(
                active_step_completed=True,
                answers_objective=True,
            ),
        ],
    )
    initial_state = create_initial_state(
        tmp_path,
        "Find the relevant implementation.",
        max_tool_calls_per_step=3,
        max_explorer_calls_per_step=3,
    )

    final_state = create_agent_graph(models).invoke(initial_state)

    assert final_state["next_action"] == "accepted"
    assert final_state["termination_status"] == "accepted"
    assert final_state["step_explorer_call_count"] == 2
    assert final_state["plan_revision_count"] == 0
    assert final_state["evaluation_report"].passed is True


def test_graph_advances_to_the_next_plan_step(tmp_path: Path) -> None:
    (tmp_path / "auth.py").write_text("value = True\n", encoding="utf-8")
    plan = InvestigationPlan(
        steps=[
            PlanStep(
                id="locate",
                objective="Locate the implementation.",
                expected_evidence=["A repository source location."],
                status="active",
            ),
            PlanStep(
                id="explain",
                objective="Explain the verified implementation.",
                expected_evidence=["Evidence-backed behavior."],
                status="pending",
            ),
        ]
    )
    expected_result = FinalResult(
        relevant_files=["auth.py"],
        evidence=["auth.py contains the relevant implementation."],
        summary="The relevant implementation is in auth.py.",
        unresolved_questions=[],
    )
    model = _create_scripted_model(
        [
            AIMessage(content="The implementation was located."),
            AIMessage(content="The implementation was explained."),
        ],
        expected_result,
    )
    models = with_scripted_planner(
        model,
        plan=plan,
        semantic_assessments=[
            _semantic_assessment(
                active_step_completed=True,
                answers_objective=True,
            ),
            _semantic_assessment(
                active_step_completed=True,
                answers_objective=True,
            ),
        ],
    )
    initial_state = create_initial_state(
        tmp_path,
        "Locate and explain the implementation.",
        max_tool_calls_per_step=3,
        max_explorer_calls_per_step=3,
    )

    final_state = create_agent_graph(models).invoke(initial_state)

    final_plan = final_state["investigation_plan"]
    assert [step.status for step in final_plan.steps] == ["completed", "active"]
    assert final_plan.active_step.id == "explain"
    assert final_state["step_explorer_call_count"] == 1
    assert final_state["plan_revision_count"] == 0
    assert final_state["next_action"] == "accepted"
    assert final_state["termination_status"] == "accepted"


def test_graph_revises_a_blocked_plan_before_retrying(tmp_path: Path) -> None:
    (tmp_path / "auth.py").write_text("value = True\n", encoding="utf-8")
    initial_plan = InvestigationPlan(
        steps=[
            PlanStep(
                id="locate",
                objective="Locate the implementation.",
                expected_evidence=["A repository source location."],
                status="active",
            )
        ]
    )
    revised_plan = InvestigationPlan(
        steps=[
            PlanStep(
                id="retry",
                objective="Try a focused implementation search.",
                expected_evidence=["A verified repository source location."],
                status="active",
            )
        ]
    )
    expected_result = FinalResult(
        relevant_files=["auth.py"],
        evidence=["auth.py contains the relevant implementation."],
        summary="The relevant implementation is in auth.py.",
        unresolved_questions=[],
    )
    model = _create_scripted_model(
        [
            AIMessage(content="The initial search was insufficient."),
            AIMessage(content="The revised search found the implementation."),
        ],
        expected_result,
    )
    models = with_scripted_planner(
        model,
        plan=initial_plan,
        revised_plans=[revised_plan],
        semantic_assessments=[
            _semantic_assessment(
                active_step_completed=False,
                answers_objective=False,
            ),
            _semantic_assessment(
                active_step_completed=True,
                answers_objective=True,
            ),
        ],
    )
    initial_state = create_initial_state(
        tmp_path,
        "Find the relevant implementation.",
        max_tool_calls_per_step=3,
        max_explorer_calls_per_step=1,
    )

    final_state = create_agent_graph(models).invoke(initial_state)

    assert final_state["investigation_plan"] == revised_plan
    assert final_state["plan_revision_count"] == 1
    assert final_state["step_explorer_call_count"] == 1
    assert final_state["next_action"] == "accepted"
    assert final_state["termination_status"] == "accepted"


def test_graph_replans_then_exhausts_repeated_invalid_tool_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gateway_module,
        "_discover_commit_sha",
        lambda _: "a" * 40,
    )
    monkeypatch.setattr(gateway_module, "_is_clean_git_worktree", lambda _: True)
    initial_plan = InvestigationPlan(
        steps=[
            PlanStep(
                id="initial-search",
                objective="Search for the implementation.",
                expected_evidence=["A verified source location."],
                status="active",
            )
        ]
    )
    revised_plan = InvestigationPlan(
        steps=[
            PlanStep(
                id="focused-search",
                objective="Retry with a focused search.",
                expected_evidence=["A verified source location."],
                status="active",
            )
        ]
    )
    invalid_call = ToolCall(
        name="list_tree",
        args={"max_depth": "not-an-int"},
        id="call-invalid",
        type="tool_call",
    )
    invalid_requests = [
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name=invalid_call["name"],
                    args=invalid_call["args"],
                    id=f"call-invalid-{index}",
                    type="tool_call",
                )
            ],
        )
        for index in range(4)
    ]
    expected_result = FinalResult(
        relevant_files=[],
        evidence=[],
        summary="Repeated invalid calls produced no repository evidence.",
        unresolved_questions=["Where is the implementation?"],
    )
    model = _create_scripted_model(invalid_requests, expected_result)
    models = with_scripted_planner(
        model,
        plan=initial_plan,
        revised_plans=[revised_plan],
        semantic_assessments=[
            _semantic_assessment(
                active_step_completed=False,
                answers_objective=False,
            ),
            _semantic_assessment(
                active_step_completed=False,
                answers_objective=False,
            ),
        ],
    )
    initial_state = create_initial_state(
        tmp_path,
        "Find the implementation.",
        max_tool_calls_per_step=8,
        max_explorer_calls_per_step=8,
        max_plan_revisions=1,
        max_no_progress_rounds=2,
    )

    final_state = create_agent_graph(models).invoke(initial_state)
    decision = decode_no_progress_decision(final_state["no_progress_decision"])

    assert final_state["investigation_plan"] == revised_plan
    assert final_state["plan_revision_count"] == 1
    assert final_state["step_tool_call_count"] == 0
    assert final_state["step_explorer_call_count"] == 2
    assert final_state["consecutive_no_progress_rounds"] == 2
    assert final_state["step_stop_reason"] == "step_no_progress"
    assert final_state["termination_status"] == "exhausted"
    assert decision.reason is NoProgressReason.NO_PROGRESS_LIMIT
    assert len(final_state["tool_audit_records"]) == 4
    assert [
        decode_tool_audit_record(payload).decision.reason
        for payload in final_state["tool_audit_records"]
    ] == [ToolDecisionReason.INVALID_ARGUMENTS, ToolDecisionReason.DUPLICATE_CALL] * 2
