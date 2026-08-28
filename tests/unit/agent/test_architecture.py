"""Executable architecture-profile and tool-wrapper baseline tests."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from repocompass.agent.architecture import (
    DEFAULT_AGENT_ARCHITECTURE,
    AgentArchitecturePreset,
    AgentArchitectureProfile,
    create_agent_architecture_profile,
)
from repocompass.agent.context_builder import ContextSelectionStrategy
from repocompass.agent.result import FinalResult
from repocompass.agent.runner import AgentRunner, RunRequest
from repocompass.agent.trace import TraceEventKind
from repocompass.models import WorkflowModels
from tests.model_doubles import candidate_from_final_result, with_scripted_planner


def _scripted_model(final_result: FinalResult) -> MagicMock:
    model = MagicMock(spec=BaseChatModel)
    model.bind_tools.return_value = RunnableLambda(
        lambda _: AIMessage(content="Enough evidence is available.")
    )
    model.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(final_result)
    )
    return model


def test_profiles_are_content_addressed_and_encode_real_feature_deltas() -> None:
    tool_wrapper = create_agent_architecture_profile(
        AgentArchitecturePreset.TOOL_WRAPPER
    )
    evaluator = create_agent_architecture_profile(
        AgentArchitecturePreset.EVALUATOR_LOOP
    )
    symbols = create_agent_architecture_profile(
        AgentArchitecturePreset.EVALUATOR_LOOP_SYMBOL_TOOLS
    )
    context = create_agent_architecture_profile(
        AgentArchitecturePreset.EVALUATOR_LOOP_CONTEXT_BUILDER
    )

    assert (
        tool_wrapper.planner_enabled,
        tool_wrapper.evaluator_enabled,
        tool_wrapper.structural_tools_enabled,
        tool_wrapper.context_selection_strategy,
    ) == (False, False, False, ContextSelectionStrategy.FIFO)
    assert evaluator.evaluator_enabled
    assert not evaluator.structural_tools_enabled
    assert evaluator.context_selection_strategy is ContextSelectionStrategy.FIFO
    assert symbols.structural_tools_enabled
    assert symbols.context_selection_strategy is ContextSelectionStrategy.FIFO
    assert context.structural_tools_enabled
    assert context.context_selection_strategy is ContextSelectionStrategy.PRIORITIZED
    assert (
        len(
            {
                tool_wrapper.profile_id,
                evaluator.profile_id,
                symbols.profile_id,
                context.profile_id,
            }
        )
        == 4
    )
    assert DEFAULT_AGENT_ARCHITECTURE == context
    assert (
        AgentArchitectureProfile.model_validate_json(context.model_dump_json())
        == context
    )

    tampered = context.model_dump(mode="json")
    tampered["evaluator_enabled"] = False
    with pytest.raises(ValidationError, match="preset|profile_id"):
        AgentArchitectureProfile.model_validate_json(json.dumps(tampered))


def test_tool_wrapper_profile_skips_planner_and_evaluator_and_binds_lexical_tools(
    tmp_path: Path,
) -> None:
    (tmp_path / "example.py").write_text("value = True\n", encoding="utf-8")
    result = FinalResult(
        relevant_files=["example.py"],
        evidence=["example.py contains the relevant implementation."],
        summary="The implementation is in example.py.",
        unresolved_questions=[],
    )
    model = _scripted_model(result)
    runner = AgentRunner(
        models=WorkflowModels.from_single_model(model),
        checkpointer=InMemorySaver(),
        architecture_profile=create_agent_architecture_profile(
            AgentArchitecturePreset.TOOL_WRAPPER
        ),
        run_id_factory=lambda: "tool-wrapper-run",
        thread_id_factory=lambda: "tool-wrapper-thread",
    )

    outcome = runner.run(
        RunRequest(
            repo_path=tmp_path,
            objective="Locate the implementation.",
            max_tool_calls_per_step=3,
            max_explorer_calls_per_step=2,
        )
    )

    nodes = [
        event.node
        for event in outcome.trace.events
        if event.kind is TraceEventKind.NODE_COMPLETED
    ]
    assert nodes == [
        "plan_investigation",
        "agent",
        "build_candidate",
        "prepare_human_review",
        "review_final_candidate",
        "mark_investigation_accepted",
        "finalize",
    ]
    assert outcome.termination_status == "accepted"
    assert outcome.final_result == result
    assert runner.architecture_profile.preset is AgentArchitecturePreset.TOOL_WRAPPER
    bound_tools = model.bind_tools.call_args.args[0]
    assert tuple(tool.name for tool in bound_tools) == (
        "list_tree",
        "search_text",
        "read_code",
    )


@pytest.mark.parametrize(
    ("preset", "expected_tool_count"),
    [
        (AgentArchitecturePreset.EVALUATOR_LOOP, 3),
        (AgentArchitecturePreset.EVALUATOR_LOOP_SYMBOL_TOOLS, 7),
        (AgentArchitecturePreset.EVALUATOR_LOOP_CONTEXT_BUILDER, 7),
    ],
)
def test_evaluator_profiles_execute_evaluator_and_expose_declared_tool_set(
    tmp_path: Path,
    preset: AgentArchitecturePreset,
    expected_tool_count: int,
) -> None:
    (tmp_path / "example.py").write_text("value = True\n", encoding="utf-8")
    result = FinalResult(
        relevant_files=["example.py"],
        evidence=["example.py contains the relevant implementation."],
        summary="The implementation is in example.py.",
        unresolved_questions=[],
    )
    explorer = _scripted_model(result)
    runner = AgentRunner(
        models=with_scripted_planner(explorer),
        checkpointer=InMemorySaver(),
        architecture_profile=create_agent_architecture_profile(preset),
    )

    outcome = runner.run(
        RunRequest(
            repo_path=tmp_path,
            objective="Locate the implementation.",
            max_tool_calls_per_step=3,
            max_explorer_calls_per_step=2,
        )
    )

    nodes = [event.node for event in outcome.trace.events]
    assert "evaluate_candidate" in nodes
    bound_tools = explorer.bind_tools.call_args.args[0]
    assert len(bound_tools) == expected_tool_count
    assert tuple(tool.name for tool in bound_tools[:3]) == (
        "list_tree",
        "search_text",
        "read_code",
    )
    assert ("list_symbols" in {tool.name for tool in bound_tools}) is (
        expected_tool_count == 7
    )
