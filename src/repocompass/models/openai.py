"""OpenAI-specific LangChain chat-model adapter."""

from langchain_core.language_models import BaseChatModel
import httpx

from repocompass.models.controlled_openai import ControlledChatOpenAI as ChatOpenAI

from repocompass.models.config import OpenAIModelSettings


def create_openai_model(settings: OpenAIModelSettings) -> BaseChatModel:
    """Create an OpenAI chat model without making an API request."""

    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        timeout=httpx.Timeout(
            settings.call_policy.request_timeout_seconds,
            connect=min(10.0, settings.call_policy.request_timeout_seconds),
        ),
        model_call_timeout=settings.call_policy.request_timeout_seconds,
        max_retries=0,
        use_responses_api=False,
        metadata={
            "repocompass_provider": "openai",
            "repocompass_model": settings.model,
        },
    )
