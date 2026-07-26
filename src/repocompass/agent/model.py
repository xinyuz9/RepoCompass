"""Construct the chat model used by the RepoCompass agent."""

import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


def create_chat_model() -> BaseChatModel:
    """Create a configured chat model without making an API request."""
    load_dotenv(override=False)

    settings = {
        "REPOCOMPASS_MODEL": os.getenv("REPOCOMPASS_MODEL", "").strip(),
        "REPOCOMPASS_BASE_URL": os.getenv("REPOCOMPASS_BASE_URL", "").strip(),
        "REPOCOMPASS_API_KEY": os.getenv("REPOCOMPASS_API_KEY", "").strip(),
    }
    missing_settings = [name for name, value in settings.items() if not value]
    if missing_settings:
        names = ", ".join(missing_settings)
        raise ValueError(f"Missing required environment variables: {names}")

    return ChatOpenAI(
        model=settings["REPOCOMPASS_MODEL"],
        api_key=settings["REPOCOMPASS_API_KEY"],
        base_url=settings["REPOCOMPASS_BASE_URL"],
        temperature=0.0,
        use_responses_api=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
