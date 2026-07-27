"""Public factories for provider clients and workflow model dependencies."""

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from repocompass.models.config import (
    DeepSeekModelSettings,
    ModelSettings,
    OpenAIModelSettings,
    load_model_settings,
)
from repocompass.models.deepseek import create_deepseek_model
from repocompass.models.openai import create_openai_model


@dataclass(frozen=True, slots=True)
class WorkflowModels:
    """Explicit model roles used by the RepoCompass workflow.

    Planner and replanner share ``planner``. Candidate construction currently
    shares ``explorer``. Keeping the roles explicit lets those choices diverge
    later without storing model clients in graph state.
    """

    explorer: BaseChatModel
    planner: BaseChatModel
    evaluator: BaseChatModel

    @classmethod
    def from_single_model(cls, model: BaseChatModel) -> "WorkflowModels":
        """Use one model client for every role during incremental migration."""

        return cls(
            explorer=model,
            planner=model,
            evaluator=model,
        )


def create_chat_model(
    settings: ModelSettings | None = None,
) -> BaseChatModel:
    """Create a provider model from explicit or environment-backed settings."""

    resolved_settings = settings or load_model_settings()

    if isinstance(resolved_settings, DeepSeekModelSettings):
        return create_deepseek_model(resolved_settings)

    if isinstance(resolved_settings, OpenAIModelSettings):
        return create_openai_model(resolved_settings)

    raise TypeError(
        f"Unsupported model settings type: {type(resolved_settings).__name__}"
    )


def create_workflow_models(
    settings: ModelSettings | None = None,
) -> WorkflowModels:
    """Create the current single-client workflow model bundle.

    Provider-specific construction remains in :func:`create_chat_model`.
    Per-role configuration can be added later without changing graph APIs.
    """

    return WorkflowModels.from_single_model(create_chat_model(settings))
