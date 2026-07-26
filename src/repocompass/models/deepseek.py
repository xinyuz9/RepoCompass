"""DeepSeek-specific LangChain chat-model adapter."""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from repocompass.models.config import DeepSeekModelSettings


def create_deepseek_model(settings: DeepSeekModelSettings) -> BaseChatModel:
    """Create a DeepSeek chat model without making an API request."""

    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0.0,
        use_responses_api=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
