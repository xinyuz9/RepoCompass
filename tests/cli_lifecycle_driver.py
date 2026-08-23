"""Deterministic model assembly for CLI lifecycle subprocess tests."""

from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from repocompass import cli
from repocompass.agent.result import FinalResult
from repocompass.models import WorkflowModels
from tests.model_doubles import candidate_from_final_result, with_scripted_planner


def _create_models() -> WorkflowModels:
    """Return process-local scripted models for one small repository fixture."""

    final_result = FinalResult(
        relevant_files=["example.py"],
        evidence=["example.py:1 contains the relevant implementation."],
        summary="The relevant implementation is in example.py.",
        unresolved_questions=[],
    )
    explorer = MagicMock(spec=BaseChatModel)
    explorer.bind_tools.return_value = RunnableLambda(
        lambda _: AIMessage(content="Enough evidence is available.")
    )
    explorer.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(final_result)
    )
    return with_scripted_planner(explorer)


def main() -> int:
    """Run the production CLI boundary with deterministic model dependencies."""

    cli.create_workflow_models = _create_models
    return cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
