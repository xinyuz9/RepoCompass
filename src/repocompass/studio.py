"""LangSmith Studio development entry point for RepoCompass."""

from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from repocompass.agent.graph import create_agent_graph
from repocompass.agent.runner import (
    DEFAULT_MAX_EXPLORER_CALLS_PER_STEP,
    DEFAULT_MAX_TOOL_CALLS_PER_STEP,
    RunRequest,
    resolve_repository_path,
)
from repocompass.agent.state import (
    RepoCompassState,
    create_initial_state,
)
from repocompass.models import WorkflowModels, create_workflow_models


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
    max_tool_calls_per_step: int = Field(
        default=DEFAULT_MAX_TOOL_CALLS_PER_STEP,
        gt=0,
        strict=True,
        description="Maximum repository tool calls per active plan step.",
    )
    max_explorer_calls_per_step: int = Field(
        default=DEFAULT_MAX_EXPLORER_CALLS_PER_STEP,
        gt=0,
        strict=True,
        description="Maximum Explorer invocations per active plan step.",
    )


class StudioState(RepoCompassState):
    """Studio-visible state containing public input and internal workflow state."""


def initialize_studio_run(state: StudioState) -> StudioState:
    """Validate Studio input and construct the internal RepoCompass state."""

    public_input = {
        field_name: state[field_name]
        for field_name in StudioRunInput.model_fields
        if field_name in state
    }
    studio_input = StudioRunInput.model_validate(public_input)
    request = RunRequest(
        repo_path=studio_input.repo_path,
        objective=studio_input.objective,
        max_tool_calls_per_step=studio_input.max_tool_calls_per_step,
        max_explorer_calls_per_step=studio_input.max_explorer_calls_per_step,
    )
    repo_path = resolve_repository_path(request.repo_path)
    initial_state = create_initial_state(
        repo_path,
        request.objective,
        request.max_tool_calls_per_step,
        request.max_explorer_calls_per_step,
    )
    return initial_state


def build_studio_graph(models: WorkflowModels):
    """Build the Studio adapter around the existing investigation graph."""

    builder = StateGraph(StudioState, input_schema=StudioRunInput)
    builder.add_node("initialize_run", initialize_studio_run)
    builder.add_node("investigate_repository", create_agent_graph(models))
    builder.add_edge(START, "initialize_run")
    builder.add_edge("initialize_run", "investigate_repository")
    builder.add_edge("investigate_repository", END)
    return builder.compile(name="repocompass_studio")


def create_studio_graph(_config: RunnableConfig):
    """Create the graph loaded by the local LangGraph Agent Server."""

    return build_studio_graph(create_workflow_models())
