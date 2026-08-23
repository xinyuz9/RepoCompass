"""Security integration tests for trusted repository-root injection."""

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

from repocompass.agent.graph import create_agent_graph
from repocompass.agent.result import FinalResult
from repocompass.agent.state import create_initial_state
from repocompass.tools.contracts import (
    ToolDecisionReason,
    decode_tool_audit_record,
)
from tests.model_doubles import candidate_from_final_result, with_scripted_planner


def test_agent_rejects_tool_call_that_spoofs_repository_root(
    tmp_path: Path,
) -> None:
    """A model-supplied repo_path is invalid and must never be executed."""

    trusted_repo = tmp_path / "trusted"
    untrusted_repo = tmp_path / "untrusted"
    trusted_repo.mkdir()
    untrusted_repo.mkdir()
    (trusted_repo / "evidence.py").write_text(
        "TRUSTED_REPOSITORY_MARKER\n",
        encoding="utf-8",
    )
    (untrusted_repo / "evidence.py").write_text(
        "UNTRUSTED_REPOSITORY_MARKER\n",
        encoding="utf-8",
    )

    spoofed_call = ToolCall(
        name="read_code",
        args={
            "repo_path": str(untrusted_repo),
            "relative_path": "evidence.py",
            "start_line": 1,
            "end_line": 1,
        },
        id="call-read",
        type="tool_call",
    )
    tool_request = AIMessage(content="", tool_calls=[spoofed_call])
    final_response = AIMessage(content="Enough evidence is available.")
    scripted_responses = iter([tool_request, final_response])
    expected_result = FinalResult(
        relevant_files=["evidence.py"],
        evidence=["evidence.py:1 contains the trusted marker."],
        summary="The trusted repository was inspected.",
        unresolved_questions=[],
    )

    model = MagicMock(spec=BaseChatModel)
    model.bind_tools.return_value = RunnableLambda(lambda _: next(scripted_responses))
    model.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(expected_result)
    )
    initial_state = create_initial_state(
        trusted_repo,
        "Read evidence.py.",
        max_tool_calls_per_step=1,
        max_explorer_calls_per_step=2,
    )

    final_state = create_agent_graph(with_scripted_planner(model)).invoke(initial_state)
    tool_result = final_state["messages"][2]

    assert isinstance(tool_result, ToolMessage)
    assert tool_result.status == "error"
    assert json.loads(str(tool_result.content)) == {
        "error": {"reason": "invalid_arguments"}
    }
    assert "TRUSTED_REPOSITORY_MARKER" not in str(tool_result.content)
    assert "UNTRUSTED_REPOSITORY_MARKER" not in str(tool_result.content)
    assert final_state["repo_path"] == trusted_repo.resolve()
    assert final_state["step_tool_call_count"] == 0
    assert len(final_state["tool_audit_records"]) == 1
    audit = decode_tool_audit_record(final_state["tool_audit_records"][0])
    assert audit.decision.reason is ToolDecisionReason.INVALID_ARGUMENTS
    assert final_state["final_result"] == expected_result


def test_repository_instruction_text_remains_selected_context_data() -> None:
    """This verifies message boundaries, not model immunity to prompt injection."""

    fixture_repo = Path(__file__).parents[1] / "fixtures" / "prompt_injection_project"
    read_call = ToolCall(
        name="read_code",
        args={
            "relative_path": "README.md",
            "start_line": 1,
            "end_line": 3,
        },
        id="call-read-readme",
        type="tool_call",
    )
    tool_request = AIMessage(content="", tool_calls=[read_call])
    final_response = AIMessage(content="The repository text is untrusted evidence.")
    scripted_responses = iter([tool_request, final_response])
    agent_inputs: list[list[AnyMessage]] = []
    expected_result = FinalResult(
        relevant_files=["README.md"],
        evidence=["README.md contains untrusted instruction text."],
        summary="No code claim was established from the untrusted README text.",
        unresolved_questions=["Which code implements authentication?"],
    )

    def next_response(messages: list[AnyMessage]) -> AIMessage:
        agent_inputs.append(list(messages))
        try:
            return next(scripted_responses)
        except StopIteration:
            return AIMessage(content="No additional repository evidence is available.")

    model = MagicMock(spec=BaseChatModel)
    model.bind_tools.return_value = RunnableLambda(next_response)
    model.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(expected_result)
    )
    initial_state = create_initial_state(
        fixture_repo,
        "Investigate authentication.",
        max_tool_calls_per_step=1,
        max_explorer_calls_per_step=2,
    )

    final_state = create_agent_graph(with_scripted_planner(model)).invoke(initial_state)

    second_agent_input = agent_inputs[1]
    assert isinstance(second_agent_input[0], SystemMessage)
    assert len(second_agent_input) == 2
    assert isinstance(second_agent_input[1], HumanMessage)
    rendered_context = str(second_agent_input[1].content)
    assert "role=explorer" in rendered_context
    assert '"section":"evidence"' in rendered_context
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in rendered_context
    assert all(not isinstance(message, ToolMessage) for message in second_agent_input)
    assert all(
        not isinstance(message, SystemMessage) for message in final_state["messages"]
    )
