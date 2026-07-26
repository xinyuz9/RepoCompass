"""Public factory that dispatches to an explicit model provider."""

from langchain_core.language_models import BaseChatModel

from repocompass.models.config import (
    DeepSeekModelSettings,
    ModelSettings,
    OpenAIModelSettings,
    load_model_settings,
)
from repocompass.models.deepseek import create_deepseek_model
from repocompass.models.openai import create_openai_model


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
