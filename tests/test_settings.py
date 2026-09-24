from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from any_llm.constants import LLMProvider
from pydantic import ValidationError

from bub.builtin.settings import AgentSettings
from bub.builtin.spill import SpillSettings
from bub.configure import Config

TEST_MODEL = "test:model"


def _settings_with_env(env: dict[str, str]) -> AgentSettings:
    with patch.dict("os.environ", env, clear=True):
        return AgentSettings(model=TEST_MODEL)


def test_settings_single_api_key_and_base() -> None:
    settings = _settings_with_env({"BUB_API_KEY": "sk-test", "BUB_API_BASE": "https://api.example.com"})

    assert isinstance(settings.api_key, str)
    assert isinstance(settings.api_base, str)


def test_settings_require_an_explicit_model() -> None:
    with patch.dict(os.environ, {}, clear=True), pytest.raises(ValidationError, match="model"):
        AgentSettings()


def test_settings_no_keys_return_none() -> None:
    settings = _settings_with_env({})

    assert settings.api_key is None
    assert settings.api_base is None
    assert settings.client_args == {}
    assert settings.completion_args == {}


@pytest.mark.parametrize("provider", ["openai", LLMProvider.OPENAI, "acme"])
def test_client_options_resolve_provider_names_and_enum_values(provider: str) -> None:
    settings = _settings_with_env({
        "BUB_API_KEY": '{"openai": "openai-key", "acme": "acme-key"}',
        "BUB_API_BASE": '{"openai": "https://api.openai.com"}',
        "BUB_CLIENT_ARGS": '{"api_key": "ignored-key", "api_base": "https://ignored.test", "timeout": 5}',
    })

    assert settings.model_client_kwargs(provider) == {
        "api_key": "openai-key" if provider != "acme" else "acme-key",
        "api_base": "https://api.openai.com" if provider != "acme" else None,
        "timeout": 5,
    }


def test_settings_load_values_from_yaml(load_config) -> None:
    with patch.dict(os.environ, {}, clear=True):
        config = load_config(
            """
model: openai:gpt-5
fallback_models:
  - openai:gpt-4o-mini
max_steps: 77
api_key:
  openai: sk-yaml
api_base:
  openai: https://api.openai.com
client_args:
  extra_headers:
    HTTP-Referer: https://openclaw.ai
    X-Title: OpenClaw
completion_args:
  reasoning_effort: high
""".strip(),
        )

        settings = config.ensure(AgentSettings)

    assert settings.model == "openai:gpt-5"
    assert settings.fallback_models == ["openai:gpt-4o-mini"]
    assert settings.max_steps == 77
    assert settings.api_key == {"openai": "sk-yaml"}
    assert settings.api_base == {"openai": "https://api.openai.com"}
    assert settings.client_args == {
        "extra_headers": {"HTTP-Referer": "https://openclaw.ai", "X-Title": "OpenClaw"},
    }
    assert settings.completion_args == {"reasoning_effort": "high"}


def test_env_settings_override_yaml(load_config) -> None:
    yaml_content = """
model: openai:gpt-5
api_key: sk-yaml
max_steps: 77
client_args:
  extra_headers:
    HTTP-Referer: https://yaml.example
    X-Title: YAML App
""".strip()

    with patch.dict(
        "os.environ",
        {
            "BUB_MODEL": "anthropic:claude-3-7-sonnet",
            "BUB_API_KEY": "sk-env",
            "BUB_CLIENT_ARGS": '{"extra_headers":{"HTTP-Referer":"https://env.example","X-Title":"Env App"}}',
            "BUB_COMPLETION_ARGS": '{"reasoning_effort":"medium"}',
            "BUB_MAX_STEPS": "12",
        },
        clear=True,
    ):
        config = load_config(yaml_content)
        settings = config.ensure(AgentSettings)

    assert settings.model == "anthropic:claude-3-7-sonnet"
    assert settings.api_key == "sk-env"
    assert settings.max_steps == 12
    assert settings.client_args == {
        "extra_headers": {"HTTP-Referer": "https://env.example", "X-Title": "Env App"},
    }
    assert settings.completion_args == {"reasoning_effort": "medium"}


def test_settings_client_args_can_be_disabled() -> None:
    settings = _settings_with_env({"BUB_CLIENT_ARGS": "null", "BUB_COMPLETION_ARGS": "null"})

    assert settings.client_args == {}
    assert settings.completion_args == {}


def test_spill_sidecar_settings_can_be_configured_or_disabled() -> None:
    with patch.dict("os.environ", {"BUB_SPILL_THRESHOLD": "64"}, clear=True):
        assert SpillSettings().threshold == 64
    with patch.dict("os.environ", {"BUB_SPILL_THRESHOLD": "0"}, clear=True):
        assert SpillSettings().threshold == 0


def test_spill_sidecar_settings_load_from_the_plugin_section(load_config) -> None:
    config = load_config("spill:\n  threshold: 64")

    assert config.ensure(SpillSettings).threshold == 64


def test_load_settings_requires_a_model_when_none_is_configured() -> None:
    with patch.dict(os.environ, {}, clear=True), pytest.raises(ValidationError, match="model"):
        Config().ensure(AgentSettings)


def test_load_settings_returns_loaded_config(load_config) -> None:
    with patch.dict(os.environ, {}, clear=True):
        config = load_config(
            """
model: openrouter:openrouter/free
""".strip(),
        )

        settings = config.ensure(AgentSettings)

    assert settings.model == "openrouter:openrouter/free"
