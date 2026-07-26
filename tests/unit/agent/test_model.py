"""Unit tests for environment-based chat model construction."""

from unittest.mock import MagicMock, sentinel

import pytest

from repocompass.agent import model as model_module

MODEL_ENVIRONMENT_VARIABLES = (
    "REPOCOMPASS_MODEL",
    "REPOCOMPASS_BASE_URL",
    "REPOCOMPASS_API_KEY",
)


def test_create_chat_model_uses_environment_without_api_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass configured values to ChatOpenAI without constructing a real client."""

    load_dotenv = MagicMock()
    chat_openai = MagicMock(return_value=sentinel.chat_model)
    monkeypatch.setattr(model_module, "load_dotenv", load_dotenv)
    monkeypatch.setattr(model_module, "ChatOpenAI", chat_openai)
    monkeypatch.setenv("REPOCOMPASS_MODEL", "test-model")
    monkeypatch.setenv("REPOCOMPASS_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("REPOCOMPASS_API_KEY", "test-key-from-environment")

    result = model_module.create_chat_model()

    assert result is sentinel.chat_model
    load_dotenv.assert_called_once_with(override=False)
    chat_openai.assert_called_once_with(
        model="test-model",
        api_key="test-key-from-environment",
        base_url="https://example.invalid/v1",
        temperature=0.0,
        use_responses_api=False,
        extra_body={"thinking": {"type": "disabled"}},
    )


def test_create_chat_model_lists_missing_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail before client construction when required settings are absent."""

    monkeypatch.setattr(model_module, "load_dotenv", MagicMock())
    chat_openai = MagicMock()
    monkeypatch.setattr(model_module, "ChatOpenAI", chat_openai)
    for variable_name in MODEL_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable_name, raising=False)

    with pytest.raises(ValueError) as error:
        model_module.create_chat_model()

    message = str(error.value)
    assert all(name in message for name in MODEL_ENVIRONMENT_VARIABLES)
    chat_openai.assert_not_called()
