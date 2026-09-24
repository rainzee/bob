from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import DemoSettings

from bub.builtin.settings import AgentSettings
from bub.configure import Config


def test_load_registers_root_and_named_config_sections(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yml"
    expected_token = "123:abc"  # noqa: S105
    config_file.write_text(
        f"""
model: openai:gpt-5
demo:
  token: {expected_token}
""".strip(),
        encoding="utf-8",
    )

    with patch.dict(os.environ, {}, clear=True):
        config = Config()
        loaded = config.load(config_file)

        assert loaded["model"] == "openai:gpt-5"
        assert loaded["demo"]["token"] == expected_token
        assert config.ensure(AgentSettings).model == "openai:gpt-5"
        assert config.ensure(DemoSettings).token == expected_token


def test_ensure_caches_within_one_config_and_not_across_instances() -> None:
    config = Config()

    assert config.ensure(AgentSettings) is config.ensure(AgentSettings)
    assert Config().ensure(AgentSettings) is not config.ensure(AgentSettings)


def test_get_value_reads_registered_section_from_yaml(load_config) -> None:
    with patch.dict(os.environ, {}, clear=True):
        config = load_config(
            """
demo:
  token: yaml-token
""".strip(),
        )

        assert config.get_value("demo.token") == "yaml-token"


def test_get_value_prefers_registered_env_over_yaml(write_config) -> None:
    config_file = write_config(
        """
demo:
  token: yaml-token
""".strip()
    )

    with patch.dict(os.environ, {"BUB_DEMO_TOKEN": "env-token"}, clear=True):
        config = Config()
        config.load(config_file)

        assert config.get_value("demo.token") == "env-token"


def test_get_value_descends_into_registered_dict_field(load_config) -> None:
    with patch.dict(os.environ, {}, clear=True):
        config = load_config(
            """
api_key:
  openai: sk-yaml
""".strip(),
        )

        assert config.get_value("api_key") == {"openai": "sk-yaml"}
        assert config.get_value("api_key.openai") == "sk-yaml"


def test_get_value_ignores_raw_unregistered_path(load_config) -> None:
    config = load_config(
        """
custom:
  nested:
    value: raw-value
""".strip(),
    )

    with pytest.raises(KeyError):
        config.get_value("custom.nested.value")


def test_get_value_returns_default_for_missing_path() -> None:
    assert Config().get_value("missing.value", default="fallback") == "fallback"
