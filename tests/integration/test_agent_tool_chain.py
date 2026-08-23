"""Deterministic integration tests for the RepoCompass M0 tool loop."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableLambda

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.graph import create_agent_graph
from repocompass.agent.result import FinalResult
from repocompass.agent.state import create_initial_state
from repocompass.tools.contracts import (
    ToolDecisionReason,
    decode_tool_audit_record,
)
from tests.model_doubles import candidate_from_final_result, with_scripted_planner

AUTH_PROJECT = Path(__file__).parents[1] / "fixtures" / "auth_project"


def test_agent_executes_three_tool_rounds_before_finalizing() -> None:
    """Run list, search, and read calls through the complete compiled graph."""

    tree_call = ToolCall(
        name="list_tree",
        args={"max_depth": 2, "max_entries": 10},
        id="call-tree",
        type="tool_call",
    )
    search_call = ToolCall(
        name="search_text",
        args={
            "query": "authenticate",
            "globs": ["*.py"],
            "max_results": 10,
        },
        id="call-search",
        type="tool_call",
    )
    read_call = ToolCall(
        name="read_code",
        args={
            "relative_path": "auth.py",
            "start_line": 1,
            "end_line": 7,
        },
        id="call-read",
        type="tool_call",
    )

    tree_request = AIMessage(content="", tool_calls=[tree_call])
    search_request = AIMessage(content="", tool_calls=[search_call])
    read_request = AIMessage(content="", tool_calls=[read_call])
    final_response = AIMessage(
        content="The authentication implementation is in auth.py."
    )
    scripted_responses = iter(
        [tree_request, search_request, read_request, final_response]
    )

    expected_result = FinalResult(
        relevant_files=["auth.py", "app.py"],
        evidence=[
            "auth.py:4 defines authenticate(username).",
            "app.py imports and calls authenticate(username).",
        ],
        summary="Authentication is implemented in auth.py and called from app.py.",
        unresolved_questions=[],
    )

    agent_inputs: list[list[AnyMessage]] = []

    def next_agent_response(messages: list[AnyMessage]) -> AIMessage:
        agent_inputs.append(list(messages))
        return next(scripted_responses)

    model = MagicMock(spec=BaseChatModel)
    model.bind_tools.return_value = RunnableLambda(next_agent_response)
    model.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(expected_result)
    )

    user_message = HumanMessage(content="Find and explain the authentication logic.")
    initial_state = create_initial_state(
        AUTH_PROJECT,
        str(user_message.content),
        max_tool_calls_per_step=3,
        max_explorer_calls_per_step=4,
    )

    final_state = create_agent_graph(with_scripted_planner(model)).invoke(initial_state)
    messages = final_state["messages"]

    assert final_state["final_result"] == expected_result
    assert final_state["evaluation_report"].passed is True
    assert final_state["step_stop_reason"] is None
    assert final_state["termination_status"] == "accepted"
    assert final_state["step_tool_call_count"] == 3
    assert final_state["step_explorer_call_count"] == 4
    assert final_state["repo_path"] == AUTH_PROJECT.resolve()
    assert len(messages) == 8
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == user_message.content
    assert messages[1] == tree_request
    assert messages[3] == search_request
    assert messages[5] == read_request
    assert messages[7] == final_response

    tree_result = messages[2]
    search_result = messages[4]
    read_result = messages[6]
    assert isinstance(tree_result, ToolMessage)
    assert isinstance(search_result, ToolMessage)
    assert isinstance(read_result, ToolMessage)
    assert tree_result.tool_call_id == "call-tree"
    assert search_result.tool_call_id == "call-search"
    assert read_result.tool_call_id == "call-read"
    tree_payload = json.loads(str(tree_result.content))
    search_payload = json.loads(str(search_result.content))
    read_payload = json.loads(str(read_result.content))
    assert {"app.py", "auth.py"}.issubset(tree_payload["entries"])
    assert any(
        match["path"] == "auth.py"
        and match["line"] == 4
        and "def authenticate" in match["text"]
        for match in search_payload["matches"]
    )
    assert read_payload["path"] == "auth.py"
    assert read_payload["start_line"] == 1
    assert read_payload["end_line"] == 7
    assert "4 | def authenticate" in read_payload["content"]

    audit_payloads = final_state["tool_audit_records"]
    audits = [decode_tool_audit_record(payload) for payload in audit_payloads]
    assert all(isinstance(payload, dict) for payload in audit_payloads)
    assert [audit.call_id for audit in audits] == [
        "call-tree",
        "call-search",
        "call-read",
    ]
    assert all(audit.decision.reason is ToolDecisionReason.ALLOWED for audit in audits)

    assert len(agent_inputs) == 4
    assert all(isinstance(messages[0], SystemMessage) for messages in agent_inputs)
    assert all(
        sum(isinstance(message, SystemMessage) for message in messages) == 1
        for messages in agent_inputs
    )
    assert all(len(messages) == 2 for messages in agent_inputs)
    assert all(isinstance(messages[1], HumanMessage) for messages in agent_inputs)
    assert all(
        not any(isinstance(message, ToolMessage) for message in messages)
        for messages in agent_inputs
    )
    assert user_message.content in str(agent_inputs[0][1].content)
    assert "role=explorer" in str(agent_inputs[0][1].content)
    assert "app.py" in str(agent_inputs[1][1].content)
    assert "auth.py" in str(agent_inputs[1][1].content)
    assert "def authenticate" in str(agent_inputs[2][1].content)
    assert "4 | def authenticate" in str(agent_inputs[3][1].content)
    assert all(
        not isinstance(message, SystemMessage) for message in final_state["messages"]
    )
    assert all(
        "repo_path" not in call["args"] for call in (tree_call, search_call, read_call)
    )
    model.bind_tools.assert_called_once()
    model.with_structured_output.assert_called_once_with(
        CandidateResult,
        method="function_calling",
        include_raw=True,
    )
