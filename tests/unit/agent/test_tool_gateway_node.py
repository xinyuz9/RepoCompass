"""Unit tests for the gateway-backed LangGraph tool execution node."""

import json
from pathlib import Path
from typing import get_type_hints

import pytest
from langchain_core.messages import AIMessage, InvalidToolCall, ToolCall, ToolMessage

import repocompass.tools.gateway as gateway_module
from repocompass.agent.nodes import (
    ToolGatewayNodeInput,
    create_execute_tools_node,
)
from repocompass.tools.contracts import (
    ToolDecisionReason,
    decode_tool_audit_record,
)


def _state(
    tmp_path: Path,
    message: AIMessage,
    *,
    used_tool_calls: int = 0,
    max_tool_calls: int = 3,
) -> ToolGatewayNodeInput:
    return {
        "messages": [message],
        "repo_path": tmp_path,
        "step_tool_call_count": used_tool_calls,
        "max_tool_calls_per_step": max_tool_calls,
        "tool_audit_records": [],
    }


def _config(run_id: str = "gateway-node-run") -> dict[str, object]:
    return {"configurable": {"repocompass_run_id": run_id}}


def test_gateway_node_declares_narrow_state_input() -> None:
    node = create_execute_tools_node()

    assert get_type_hints(node)["state"] is ToolGatewayNodeInput


def test_gateway_node_executes_and_audits_one_valid_call(tmp_path: Path) -> None:
    (tmp_path / "auth.py").write_text("enabled = True\n", encoding="utf-8")
    message = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="list_tree",
                args={"max_depth": 1, "max_entries": 10},
                id="call-tree",
                type="tool_call",
            )
        ],
    )

    update = create_execute_tools_node()(_state(tmp_path, message), _config())

    assert update["step_tool_call_count"] == 1
    assert update["step_stop_reason"] is None
    assert len(update["messages"]) == 1
    tool_message = update["messages"][0]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.status == "success"
    assert tool_message.tool_call_id == "call-tree"
    assert '"entries":["auth.py"]' in str(tool_message.content)
    assert len(update["tool_audit_records"]) == 1
    audit_payload = update["tool_audit_records"][0]
    audit = decode_tool_audit_record(audit_payload)
    assert audit.run_id == "gateway-node-run"
    assert audit.call_id == "call-tree"
    assert audit.decision.reason is ToolDecisionReason.ALLOWED
    assert "auth.py" not in audit.model_dump_json()
    assert isinstance(audit_payload, dict)


def test_gateway_node_reuses_equivalent_success_without_incrementing_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gateway_module,
        "_discover_commit_sha",
        lambda _root: "a" * 40,
    )
    monkeypatch.setattr(
        gateway_module,
        "_is_clean_git_worktree",
        lambda _root: True,
    )
    (tmp_path / "auth.py").write_text("enabled = True\n", encoding="utf-8")
    first_message = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="list_tree",
                args={"max_depth": 1, "max_entries": 10},
                id="call-first",
                type="tool_call",
            )
        ],
    )
    node = create_execute_tools_node()
    first_update = node(_state(tmp_path, first_message), _config("cache-node-run"))
    second_message = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="list_tree",
                args={"max_entries": 10, "max_depth": 1},
                id="call-second",
                type="tool_call",
            )
        ],
    )
    second_state = _state(
        tmp_path,
        second_message,
        used_tool_calls=first_update["step_tool_call_count"],
    )
    second_state["tool_audit_records"] = first_update["tool_audit_records"]

    second_update = node(second_state, _config("cache-node-run"))

    assert first_update["step_tool_call_count"] == 1
    assert second_update["step_tool_call_count"] == 1
    assert second_update["messages"][0].status == "success"
    assert second_update["messages"][0].content == first_update["messages"][0].content
    assert len(second_update["tool_audit_records"]) == 1
    cached_audit = decode_tool_audit_record(second_update["tool_audit_records"][0])
    assert cached_audit.sequence == 1
    assert cached_audit.call_id == "call-second"
    assert cached_audit.decision.reason is ToolDecisionReason.CACHE_HIT


def test_gateway_node_rejects_and_audits_entire_mixed_invalid_batch(
    tmp_path: Path,
) -> None:
    raw_invalid_arguments = '{"query":"authorization=Bearer secret"'
    message = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="list_tree",
                args={"max_depth": 1},
                id="call-tree",
                type="tool_call",
            )
        ],
        invalid_tool_calls=[
            InvalidToolCall(
                name="search_text",
                args=raw_invalid_arguments,
                id="call-invalid-search",
                error="private parser details",
                type="invalid_tool_call",
            )
        ],
    )

    update = create_execute_tools_node()(_state(tmp_path, message), _config())

    assert update["step_tool_call_count"] == 0
    assert update["step_stop_reason"] is None
    assert [tool_message.tool_call_id for tool_message in update["messages"]] == [
        "call-invalid-search",
        "call-tree",
    ]
    assert all(tool_message.status == "error" for tool_message in update["messages"])
    assert [
        decode_tool_audit_record(record).decision.reason
        for record in update["tool_audit_records"]
    ] == [
        ToolDecisionReason.INVALID_ARGUMENTS,
        ToolDecisionReason.INVALID_ARGUMENTS,
    ]
    serialized_update = "".join(
        json.dumps(record, sort_keys=True) for record in update["tool_audit_records"]
    )
    serialized_messages = "".join(
        str(tool_message.content) for tool_message in update["messages"]
    )
    assert "Bearer secret" not in serialized_update
    assert "private parser details" not in serialized_messages


def test_gateway_node_atomically_rejects_batch_exceeding_remaining_budget(
    tmp_path: Path,
) -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="list_tree",
                args={},
                id="call-tree",
                type="tool_call",
            ),
            ToolCall(
                name="search_text",
                args={"query": "auth"},
                id="call-search",
                type="tool_call",
            ),
        ],
    )

    update = create_execute_tools_node()(
        _state(tmp_path, message, used_tool_calls=1, max_tool_calls=2),
        _config(),
    )

    assert update["step_tool_call_count"] == 1
    assert update["step_stop_reason"] == "step_tool_call_budget_exhausted"
    assert all(tool_message.status == "error" for tool_message in update["messages"])
    assert [
        decode_tool_audit_record(record).decision.reason
        for record in update["tool_audit_records"]
    ] == [
        ToolDecisionReason.BUDGET_EXHAUSTED,
        ToolDecisionReason.BUDGET_EXHAUSTED,
    ]
    assert all(
        '"budget_exhausted"' in str(tool_message.content)
        for tool_message in update["messages"]
    )


def test_gateway_node_uses_prior_attempt_records_only_for_state_history(
    tmp_path: Path,
) -> None:
    first_message = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="list_tree",
                args={},
                id="call-first",
                type="tool_call",
            )
        ],
    )
    node = create_execute_tools_node()
    first_update = node(_state(tmp_path, first_message), _config("first-run"))
    second_message = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="list_tree",
                args={},
                id="call-second",
                type="tool_call",
            )
        ],
    )
    second_state = _state(tmp_path, second_message)
    second_state["tool_audit_records"] = first_update["tool_audit_records"]

    second_update = node(second_state, _config("second-run"))

    assert len(second_update["tool_audit_records"]) == 1
    second_audit = decode_tool_audit_record(second_update["tool_audit_records"][0])
    assert second_audit.run_id == "second-run"
    assert second_audit.sequence == 0


def test_gateway_node_uses_stable_fallback_run_id_for_direct_graph_calls(
    tmp_path: Path,
) -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="list_tree",
                args={},
                id="call-tree",
                type="tool_call",
            )
        ],
    )

    update = create_execute_tools_node()(_state(tmp_path, message), {})

    assert (
        decode_tool_audit_record(update["tool_audit_records"][0]).run_id
        == "graph-invoke"
    )
