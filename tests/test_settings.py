from __future__ import annotations

from typing import Any

import pytest
from any_llm.constants import LLMProvider
from pydantic import ValidationError

from bub.builtin.settings import AgentSettings
from bub.builtin.spill import SpillSettings
from bub.configure import Config

MODEL = "openai:gpt-5"


def _settings(data: dict[str, Any] | None = None) -> AgentSettings:
    return Config({"model": MODEL, **(data or {})}).ensure(AgentSettings)


def test_settings_single_api_key_and_base() -> None:
    settings = _settings({"api_key": "sk-test", "api_base": "https://api.example.com"})

    assert settings.api_key == "sk-test"
    assert settings.api_base == "https://api.example.com"


def test_settings_require_an_explicit_model() -> None:
    with pytest.raises(ValidationError, match="model"):
        AgentSettings()


def test_settings_without_keys_default_to_none() -> None:
    settings = _settings()

    assert settings.api_key is None
    assert settings.api_base is None
    assert settings.client_args == {}
    assert settings.completion_args == {}


@pytest.mark.parametrize("provider", ["openai", LLMProvider.OPENAI, "acme"])
def test_client_options_resolve_provider_names_and_enum_values(provider: str) -> None:
    settings = _settings({
        "api_key": {"openai": "openai-key", "acme": "acme-key"},
        "api_base": {"openai": "https://api.openai.com"},
        "client_args": {"api_key": "ignored-key", "api_base": "https://ignored.test", "timeout": 5},
    })

    assert settings.model_client_kwargs(provider) == {
        "api_key": "openai-key" if provider != "acme" else "acme-key",
        "api_base": "https://api.openai.com" if provider != "acme" else None,
        "timeout": 5,
    }


def test_settings_load_values_from_config_data() -> None:
    settings = _settings({
        "fallback_models": ["openai:gpt-4o-mini"],
        "max_steps": 77,
        "api_key": {"openai": "sk-yaml"},
        "api_base": {"openai": "https://api.openai.com"},
        "client_args": {"extra_headers": {"HTTP-Referer": "https://yaml.example", "X-Title": "YAML App"}},
        "completion_args": {"reasoning_effort": "high"},
    })

    assert settings.model == MODEL
    assert settings.fallback_models == ["openai:gpt-4o-mini"]
    assert settings.max_steps == 77
    assert settings.api_key == {"openai": "sk-yaml"}
    assert settings.api_base == {"openai": "https://api.openai.com"}
    assert settings.client_args == {
        "extra_headers": {"HTTP-Referer": "https://yaml.example", "X-Title": "YAML App"},
    }
    assert settings.completion_args == {"reasoning_effort": "high"}


def test_settings_client_args_can_be_null() -> None:
    settings = _settings({"client_args": None, "completion_args": None})

    assert settings.client_args == {}
    assert settings.completion_args == {}


def test_model_options_are_applied_to_every_candidate() -> None:
    settings = _settings({"fallback_models": ["anthropic:claude-3"], "max_tokens": 512})

    assert [candidate.name for candidate in settings.model_candidates(MODEL)] == [MODEL, "anthropic:claude-3"]
    assert settings.max_tokens == 512


def test_max_tokens_falls_back_to_the_default() -> None:
    from bub.builtin.settings import DEFAULT_MAX_TOKENS

    assert _settings().max_tokens == DEFAULT_MAX_TOKENS


def test_spill_sidecar_settings_load_from_the_plugin_section(write_config) -> None:
    config = Config.from_file(write_config("spill:\n  threshold: 64"))

    assert config.ensure(SpillSettings).threshold == 64


def test_spill_sidecar_settings_can_be_disabled() -> None:
    assert Config({"spill": {"threshold": 0}}).ensure(SpillSettings).threshold == 0
    assert SpillSettings().threshold == 4096


def test_load_settings_reports_a_missing_model() -> None:
    with pytest.raises(ValidationError, match="model"):
        Config().ensure(AgentSettings)


def test_load_settings_returns_loaded_config() -> None:
    config = Config({"model": "openrouter:openrouter/free"})

    assert config.ensure(AgentSettings).model == "openrouter:openrouter/free"
