"""DeepSeek-specific LangChain chat-model adapter."""

from langchain_core.language_models import BaseChatModel
import httpx

from repocompass.models.controlled_openai import ControlledChatOpenAI as ChatOpenAI

from repocompass.models.config import DeepSeekModelSettings


def create_deepseek_model(settings: DeepSeekModelSettings) -> BaseChatModel:
    """Create a DeepSeek chat model without making an API request."""

    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        timeout=httpx.Timeout(
            settings.call_policy.request_timeout_seconds,
            connect=min(10.0, settings.call_policy.request_timeout_seconds),
        ),
        model_call_timeout=settings.call_policy.request_timeout_seconds,
        max_retries=0,
        base_url=settings.base_url,
        temperature=0.0,
        use_responses_api=False,
        extra_body={"thinking": {"type": "disabled"}},
        metadata={
            "repocompass_provider": "deepseek",
            "repocompass_model": settings.model,
        },
    )
