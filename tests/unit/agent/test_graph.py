"""Unit tests for the compiled RepoCompass agent graph."""

from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    InvalidToolCall,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableLambda

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.graph import create_agent_graph
from repocompass.agent.result import FinalResult
from repocompass.agent.state import create_initial_state
from tests.model_doubles import candidate_from_final_result, with_scripted_planner


def _create_scripted_model(
    agent_responses: list[AIMessage],
    final_result: FinalResult,
) -> MagicMock:
    """Create a model double with deterministic Explorer and candidate responses."""

    response_iterator = iter(agent_responses)
    model = MagicMock(spec=BaseChatModel)
    model.bind_tools.return_value = RunnableLambda(lambda _: next(response_iterator))
    model.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(final_result)
    )
    return model


def test_graph_finalizes_when_agent_returns_no_tool_calls(tmp_path: Path) -> None:
    """A tool-free agent response should be finalized and terminate normally."""

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

    assert final_state["final_result"] == expected_result
    assert final_state["candidate_result"] == candidate_from_final_result(
        expected_result
    )
    assert final_state["stop_reason"] is None
    assert final_state["step_tool_call_count"] == 0
    assert final_state["step_explorer_call_count"] == 1
    messages = final_state["messages"]
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == user_message.content
    assert messages[1] == agent_response
    model.bind_tools.assert_called_once()
    model.with_structured_output.assert_called_once_with(
        CandidateResult,
        method="function_calling",
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

    assert final_state["final_result"] == expected_result
    assert final_state["evaluation_report"].passed is True
    assert final_state["stop_reason"] is None
    assert final_state["step_tool_call_count"] == 1
    assert final_state["step_explorer_call_count"] == 2
    assert len(messages) == 4
    assert messages[1] == tool_request
    assert isinstance(messages[2], ToolMessage)
    assert messages[3] == enough_evidence_response
    assert messages[2].tool_call_id == "call-tree"
    assert "auth.py" in str(messages[2].content)
    assert "repo_path" not in tree_call["args"]
    model.bind_tools.assert_called_once()
    model.with_structured_output.assert_called_once_with(
        CandidateResult,
        method="function_calling",
    )


def test_graph_rejects_batch_exceeding_budget_and_finalizes(
    tmp_path: Path,
) -> None:
    """An oversized batch should be rejected without consuming or executing it."""

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

    assert final_state["final_result"] == expected_result
    assert final_state["stop_reason"] == "step_tool_call_budget_exhausted"
    assert final_state["step_tool_call_count"] == 0
    assert final_state["step_explorer_call_count"] == 1
    assert len(messages) == 4
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
    assert "budget was exhausted" in str(first_rejection.content)
    assert "budget was exhausted" in str(second_rejection.content)


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
        relevant_files=[],
        evidence=[],
        summary="No repository evidence was collected from the rejected batch.",
        unresolved_questions=["Which files contain the authentication logic?"],
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

    assert final_state["final_result"] == expected_result
    assert final_state["stop_reason"] is None
    assert final_state["step_tool_call_count"] == 2
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
    assert "Invalid JSON arguments" in str(invalid_error.content)
    assert withheld_valid_call.tool_call_id == "call-tree"
    assert withheld_valid_call.status == "error"
    assert "another call in the same batch" in str(withheld_valid_call.content)
    assert "auth.py" not in str(withheld_valid_call.content)
    assert messages[4] == corrected_response


def test_graph_returns_recoverable_tool_invocation_error_to_agent(
    tmp_path: Path,
) -> None:
    """A schema error should become a ToolMessage before Agent runs again."""

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
        relevant_files=[],
        evidence=[],
        summary="The invalid tool request was handled without ending the graph.",
        unresolved_questions=["Which repository files are relevant?"],
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

    assert final_state["final_result"] == expected_result
    assert final_state["stop_reason"] is None
    assert final_state["step_tool_call_count"] == 1
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
    assert "Error invoking tool 'list_tree'" in str(tool_error.content)
    assert "max_depth" in str(tool_error.content)
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
    assert final_state["stop_reason"] == "step_explorer_budget_exhausted"
    assert final_state["final_result"] == expected_result
    assert isinstance(final_state["messages"][-1], ToolMessage)
