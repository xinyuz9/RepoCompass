from pathlib import Path

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState

from repocompass.agent.result import FinalResult


class AgentState(MessagesState):
    repo_path: Path
    tool_call_count: int
    max_tool_calls: int
    iteration_count: int
    max_iterations: int
    stop_reason: str | None
    final_result: FinalResult | None


def create_initial_state(
    repo_path: Path,
    messages: list[AnyMessage],
    max_tool_calls: int,
    max_iterations: int,
) -> AgentState:
    if max_tool_calls < 1:
        raise ValueError(f"Max tool calls must be positive, not {max_tool_calls}")
    if max_iterations < 1:
        raise ValueError(f"Max iterations must be positive, not {max_iterations}")

    return {
        "messages": list(messages),
        "repo_path": repo_path.resolve(),
        "tool_call_count": 0,
        "max_tool_calls": max_tool_calls,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "stop_reason": None,
        "final_result": None,
    }
