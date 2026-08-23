"""Unit tests for the LangSmith Studio development adapter."""

from pathlib import Path
from unittest.mock import MagicMock, sentinel

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from repocompass import studio
from repocompass.agent.lifecycle import (
    HumanDecision,
    HumanDecisionAction,
    HumanReviewPolicy,
    decode_human_review_interrupt,
    encode_human_decision,
)
from repocompass.agent.runner import (
    DEFAULT_MAX_EXPLORER_CALLS_PER_STEP,
    DEFAULT_MAX_PLAN_REVISIONS,
    DEFAULT_MAX_TOOL_CALLS_PER_STEP,
)
from repocompass.agent.result import FinalResult
from tests.model_doubles import candidate_from_final_result, with_scripted_planner


def _create_scripted_model(final_result: FinalResult) -> MagicMock:
    model = MagicMock(spec=BaseChatModel)
    model.bind_tools.return_value = RunnableLambda(
        lambda _: AIMessage(content="Enough evidence is available.")
    )
    model.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(final_result)
    )
    return model


def test_studio_graph_accepts_public_input_and_applies_defaults(
    tmp_path: Path,
) -> None:
    (tmp_path / "example.py").write_text("value = True\n", encoding="utf-8")
    final_result = FinalResult(
        relevant_files=["example.py"],
        evidence=["example.py contains the relevant implementation."],
        summary="The relevant implementation is in example.py.",
        unresolved_questions=[],
    )
    graph = studio.build_studio_graph(
        with_scripted_planner(_create_scripted_model(final_result))
    )

    final_state = graph.invoke(
        {
            "repo_path": str(tmp_path),
            "objective": "  Inspect the source package.  ",
        }
    )

    assert final_state["repo_path"] == tmp_path.resolve()
    assert final_state["objective"] == "Inspect the source package."
    assert final_state["max_tool_calls_per_step"] == DEFAULT_MAX_TOOL_CALLS_PER_STEP
    assert (
        final_state["max_explorer_calls_per_step"]
        == DEFAULT_MAX_EXPLORER_CALLS_PER_STEP
    )
    assert final_state["max_plan_revisions"] == DEFAULT_MAX_PLAN_REVISIONS
    assert final_state["human_review_policy"] is HumanReviewPolicy.NEVER
    assert final_state["plan_revision_count"] == 0
    assert final_state["step_tool_call_count"] == 0
    assert final_state["step_explorer_call_count"] == 1
    assert final_state["evaluation_report"] is not None
    assert final_state["final_result"] == final_result
    first_message = final_state["messages"][0]
    assert isinstance(first_message, HumanMessage)
    assert first_message.content == "Inspect the source package."


def test_studio_graph_exposes_only_public_input_fields() -> None:
    final_result = FinalResult(
        relevant_files=[],
        evidence=[],
        summary="No repository files were needed.",
        unresolved_questions=[],
    )
    graph = studio.build_studio_graph(
        with_scripted_planner(_create_scripted_model(final_result))
    )

    schema = graph.input_schema.model_json_schema()

    assert set(schema["properties"]) == {
        "repo_path",
        "objective",
        "max_tool_calls_per_step",
        "max_explorer_calls_per_step",
        "max_plan_revisions",
        "human_review_policy",
    }
    assert schema["properties"]["max_tool_calls_per_step"]["default"] == 16
    assert (
        schema["properties"]["max_explorer_calls_per_step"]["default"]
        == DEFAULT_MAX_EXPLORER_CALLS_PER_STEP
    )
    assert schema["properties"]["max_plan_revisions"]["default"] == 2
    assert schema["properties"]["human_review_policy"]["default"] == "never"


def test_studio_graph_surfaces_and_resumes_native_human_review(
    tmp_path: Path,
) -> None:
    (tmp_path / "example.py").write_text("value = True\n", encoding="utf-8")
    final_result = FinalResult(
        relevant_files=["example.py"],
        evidence=["example.py contains the relevant implementation."],
        summary="The relevant implementation is in example.py.",
        unresolved_questions=[],
    )
    graph = studio.build_studio_graph(
        with_scripted_planner(_create_scripted_model(final_result)),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "studio-review-thread"}}

    interrupted = graph.invoke(
        {
            "repo_path": str(tmp_path),
            "objective": "Inspect the source package.",
            "human_review_policy": "before_finalize",
        },
        config=config,
        version="v2",
    )

    assert len(interrupted.interrupts) == 1
    review_request = decode_human_review_interrupt(interrupted.interrupts[0].value)
    assert review_request.candidate_result == candidate_from_final_result(final_result)
    assert interrupted.value["termination_status"] is None
    assert interrupted.value["final_result"] is None

    resumed = graph.invoke(
        Command(
            resume=encode_human_decision(
                HumanDecision(action=HumanDecisionAction.APPROVE)
            )
        ),
        config=config,
        version="v2",
    )

    assert resumed.interrupts == ()
    assert resumed.value["termination_status"] == "accepted"
    assert resumed.value["final_result"] == final_result


def test_studio_graph_rejects_file_as_repository_path(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-directory.py"
    file_path.write_text("value = 1\n", encoding="utf-8")
    final_result = FinalResult(
        relevant_files=[],
        evidence=[],
        summary="Unused.",
        unresolved_questions=[],
    )
    graph = studio.build_studio_graph(
        with_scripted_planner(_create_scripted_model(final_result))
    )

    with pytest.raises(ValueError, match="not a directory"):
        graph.invoke(
            {
                "repo_path": str(file_path),
                "objective": "Inspect the source package.",
            }
        )


def test_agent_server_factory_uses_configured_provider_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_workflow_models = MagicMock(return_value=sentinel.models)
    build_studio_graph = MagicMock(return_value=sentinel.graph)
    monkeypatch.setattr(studio, "create_workflow_models", create_workflow_models)
    monkeypatch.setattr(studio, "build_studio_graph", build_studio_graph)

    result = studio.create_studio_graph({})

    assert result is sentinel.graph
    create_workflow_models.assert_called_once_with()
    build_studio_graph.assert_called_once_with(sentinel.models)
