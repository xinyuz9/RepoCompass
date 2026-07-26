"""OpenAI-specific LangChain chat-model adapter."""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from repocompass.models.config import OpenAIModelSettings


def create_openai_model(settings: OpenAIModelSettings) -> BaseChatModel:
    """Create an OpenAI chat model without making an API request."""

    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        use_responses_api=False,
    )
