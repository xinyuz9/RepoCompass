"""Environment-backed configuration for supported model providers."""

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

from dotenv import load_dotenv


class ModelProvider(StrEnum):
    """Model providers with an implemented RepoCompass adapter."""

    DEEPSEEK = "deepseek"
    OPENAI = "openai"


@dataclass(frozen=True, slots=True)
class DeepSeekModelSettings:
    """Validated settings consumed only by the DeepSeek adapter."""

    model: str
    api_key: str = field(repr=False)
    base_url: str
    provider: ClassVar[ModelProvider] = ModelProvider.DEEPSEEK


@dataclass(frozen=True, slots=True)
class OpenAIModelSettings:
    """Validated settings consumed only by the OpenAI adapter."""

    model: str
    api_key: str = field(repr=False)
    provider: ClassVar[ModelProvider] = ModelProvider.OPENAI


type ModelSettings = DeepSeekModelSettings | OpenAIModelSettings


def _required_environment(*names: str) -> dict[str, str]:
    """Read required non-empty environment variables."""

    values = {name: os.getenv(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
    return values


def load_model_settings() -> ModelSettings:
    """Load one provider-specific settings object without creating a client."""

    load_dotenv(override=False)
    provider_name = (
        os.getenv("REPOCOMPASS_MODEL_PROVIDER", ModelProvider.DEEPSEEK).strip().lower()
    )
    try:
        provider = ModelProvider(provider_name)
    except ValueError as error:
        supported = ", ".join(provider.value for provider in ModelProvider)
        raise ValueError(
            f"Unsupported REPOCOMPASS_MODEL_PROVIDER {provider_name!r}; "
            f"expected one of: {supported}"
        ) from error

    if provider is ModelProvider.DEEPSEEK:
        values = _required_environment(
            "REPOCOMPASS_MODEL",
            "REPOCOMPASS_API_KEY",
            "REPOCOMPASS_BASE_URL",
        )
        return DeepSeekModelSettings(
            model=values["REPOCOMPASS_MODEL"],
            api_key=values["REPOCOMPASS_API_KEY"],
            base_url=values["REPOCOMPASS_BASE_URL"],
        )

    values = _required_environment(
        "REPOCOMPASS_MODEL",
        "REPOCOMPASS_API_KEY",
    )
    return OpenAIModelSettings(
        model=values["REPOCOMPASS_MODEL"],
        api_key=values["REPOCOMPASS_API_KEY"],
    )
