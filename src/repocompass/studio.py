"""LangSmith Studio development entry point for RepoCompass."""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from repocompass.agent.graph import create_agent_graph
from repocompass.agent.runner import RunRequest, resolve_repository_path
from repocompass.agent.state import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TOOL_CALLS,
    AgentState,
    create_initial_state,
)
from repocompass.models import create_chat_model


class StudioRunInput(BaseModel):
    """Validated public input rendered by LangSmith Studio."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    repo_path: Path = Field(
        description="Trusted local source or repository directory to investigate."
    )
    objective: str = Field(
        min_length=1,
        description="Repository investigation question.",
    )
    max_tool_calls: int = Field(
        default=DEFAULT_MAX_TOOL_CALLS,
        gt=0,
        strict=True,
        description="Maximum repository tool calls allowed.",
    )
    max_iterations: int = Field(
        default=DEFAULT_MAX_ITERATIONS,
        gt=0,
        strict=True,
        description="Maximum Agent model invocations allowed.",
    )


class StudioState(AgentState):
    """Studio-visible state containing public input and internal Agent state."""

    objective: str


def initialize_studio_run(state: StudioState) -> StudioState:
    """Validate Studio input and construct the existing internal Agent state."""

    public_input = {
        field_name: state[field_name]
        for field_name in StudioRunInput.model_fields
        if field_name in state
    }
    studio_input = StudioRunInput.model_validate(public_input)
    request = RunRequest(
        repo_path=studio_input.repo_path,
        objective=studio_input.objective,
        max_tool_calls=studio_input.max_tool_calls,
        max_iterations=studio_input.max_iterations,
    )
    repo_path = resolve_repository_path(request.repo_path)
    initial_state = create_initial_state(
        repo_path,
        [HumanMessage(content=request.objective)],
        request.max_tool_calls,
        request.max_iterations,
    )
    return {
        "objective": request.objective,
        **initial_state,
    }


def build_studio_graph(model: BaseChatModel):
    """Build the Studio adapter around the existing investigation graph."""

    builder = StateGraph(StudioState, input_schema=StudioRunInput)
    builder.add_node("initialize_run", initialize_studio_run)
    builder.add_node("investigate_repository", create_agent_graph(model))
    builder.add_edge(START, "initialize_run")
    builder.add_edge("initialize_run", "investigate_repository")
    builder.add_edge("investigate_repository", END)
    return builder.compile(name="repocompass_studio")


def create_studio_graph(_config: RunnableConfig):
    """Create the graph loaded by the local LangGraph Agent Server."""

    return build_studio_graph(create_chat_model())
