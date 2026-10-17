"""Unit tests for provider-specific model configuration and construction."""

from unittest.mock import MagicMock, sentinel

import pytest
import httpx

from repocompass.models import config, deepseek, factory, openai
from repocompass.models.config import (
    DeepSeekModelSettings,
    ModelProvider,
    OpenAIModelSettings,
)
from repocompass.models.factory import WorkflowModels

MODEL_ENVIRONMENT_VARIABLES = (
    "REPOCOMPASS_MODEL_PROVIDER",
    "REPOCOMPASS_MODEL",
    "REPOCOMPASS_BASE_URL",
    "REPOCOMPASS_API_KEY",
    "REPOCOMPASS_MODEL_TIMEOUT_SECONDS",
    "REPOCOMPASS_ATTEMPT_TIMEOUT_SECONDS",
)


def _clear_model_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable_name in MODEL_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable_name, raising=False)


def test_load_model_settings_defaults_to_deepseek(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_dotenv = MagicMock()
    monkeypatch.setattr(config, "load_dotenv", load_dotenv)
    _clear_model_environment(monkeypatch)
    monkeypatch.setenv("REPOCOMPASS_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("REPOCOMPASS_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("REPOCOMPASS_API_KEY", "deepseek-test-key")

    settings = config.load_model_settings()

    assert settings == DeepSeekModelSettings(
        model="deepseek-v4-flash",
        api_key="deepseek-test-key",
        base_url="https://api.deepseek.com",
    )
    assert settings.provider is ModelProvider.DEEPSEEK
    assert "deepseek-test-key" not in repr(settings)
    load_dotenv.assert_called_once_with(override=False)


def test_load_model_settings_selects_openai_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "load_dotenv", MagicMock())
    _clear_model_environment(monkeypatch)
    monkeypatch.setenv("REPOCOMPASS_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("REPOCOMPASS_MODEL", "test-openai-model")
    monkeypatch.setenv("REPOCOMPASS_API_KEY", "openai-test-key")

    settings = config.load_model_settings()

    assert settings == OpenAIModelSettings(
        model="test-openai-model",
        api_key="openai-test-key",
    )
    assert settings.provider is ModelProvider.OPENAI
    assert "openai-test-key" not in repr(settings)


def test_load_model_settings_rejects_unknown_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "load_dotenv", MagicMock())
    _clear_model_environment(monkeypatch)
    monkeypatch.setenv("REPOCOMPASS_MODEL_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="Unsupported REPOCOMPASS_MODEL_PROVIDER"):
        config.load_model_settings()


def test_load_deepseek_settings_requires_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "load_dotenv", MagicMock())
    _clear_model_environment(monkeypatch)
    monkeypatch.setenv("REPOCOMPASS_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("REPOCOMPASS_API_KEY", "deepseek-test-key")

    with pytest.raises(ValueError, match="REPOCOMPASS_BASE_URL"):
        config.load_model_settings()


def test_create_deepseek_model_keeps_provider_parameters_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_openai = MagicMock(return_value=sentinel.deepseek_model)
    monkeypatch.setattr(deepseek, "ChatOpenAI", chat_openai)
    settings = DeepSeekModelSettings(
        model="deepseek-v4-flash",
        api_key="deepseek-test-key",
        base_url="https://api.deepseek.com",
    )

    result = deepseek.create_deepseek_model(settings)

    assert result is sentinel.deepseek_model
    chat_openai.assert_called_once_with(
        model="deepseek-v4-flash",
        api_key="deepseek-test-key",
        base_url="https://api.deepseek.com",
        temperature=0.0,
        timeout=httpx.Timeout(60.0, connect=10.0),
        model_call_timeout=60.0,
        max_retries=0,
        use_responses_api=False,
        extra_body={"thinking": {"type": "disabled"}},
        metadata={
            "repocompass_provider": "deepseek",
            "repocompass_model": "deepseek-v4-flash",
        },
    )


def test_create_openai_model_excludes_deepseek_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_openai = MagicMock(return_value=sentinel.openai_model)
    monkeypatch.setattr(openai, "ChatOpenAI", chat_openai)
    settings = OpenAIModelSettings(
        model="test-openai-model",
        api_key="openai-test-key",
    )

    result = openai.create_openai_model(settings)

    assert result is sentinel.openai_model
    chat_openai.assert_called_once_with(
        model="test-openai-model",
        api_key="openai-test-key",
        use_responses_api=False,
        timeout=httpx.Timeout(60.0, connect=10.0),
        model_call_timeout=60.0,
        max_retries=0,
        metadata={
            "repocompass_provider": "openai",
            "repocompass_model": "test-openai-model",
        },
    )


@pytest.mark.parametrize(
    ("settings", "factory_name", "expected"),
    [
        (
            DeepSeekModelSettings(
                model="deepseek-v4-flash",
                api_key="deepseek-test-key",
                base_url="https://api.deepseek.com",
            ),
            "create_deepseek_model",
            sentinel.deepseek_model,
        ),
        (
            OpenAIModelSettings(
                model="test-openai-model",
                api_key="openai-test-key",
            ),
            "create_openai_model",
            sentinel.openai_model,
        ),
    ],
)
def test_create_chat_model_dispatches_by_settings_type(
    settings: DeepSeekModelSettings | OpenAIModelSettings,
    factory_name: str,
    expected: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_factory = MagicMock(return_value=expected)
    monkeypatch.setattr(factory, factory_name, provider_factory)

    result = factory.create_chat_model(settings)

    assert result is expected
    provider_factory.assert_called_once_with(settings)


def test_workflow_models_from_single_model_preserves_explicit_roles() -> None:
    model = MagicMock()

    models = WorkflowModels.from_single_model(model)

    assert models.explorer is model
    assert models.planner is model
    assert models.evaluator is model


def test_create_workflow_models_constructs_one_shared_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = OpenAIModelSettings(
        model="test-openai-model",
        api_key="openai-test-key",
    )
    model = MagicMock()
    create_chat_model = MagicMock(return_value=model)
    monkeypatch.setattr(factory, "create_chat_model", create_chat_model)

    models = factory.create_workflow_models(settings)

    assert models == WorkflowModels.from_single_model(model)
    create_chat_model.assert_called_once_with(settings)


@pytest.mark.parametrize("provider", ["openai", "deepseek"])
def test_real_adapters_enforce_finite_timeouts_and_disable_sdk_retries(provider):
    from repocompass.models.control import ModelCallPolicy

    policy = ModelCallPolicy(2.5, 30)
    if provider == "openai":
        settings = OpenAIModelSettings(
            model="offline", api_key="not-a-key", call_policy=policy
        )
    else:
        settings = DeepSeekModelSettings(
            model="offline",
            api_key="not-a-key",
            base_url="http://127.0.0.1:1/v1",
            call_policy=policy,
        )
    models = factory.create_workflow_models(settings)
    assert models.call_policy == policy
    assert models.explorer.max_retries == 0
    assert models.explorer.root_client.max_retries == 0
    assert models.explorer.root_async_client.max_retries == 0
    assert models.explorer.request_timeout == httpx.Timeout(2.5, connect=2.5)
    assert models.explorer.model_call_timeout == 2.5
