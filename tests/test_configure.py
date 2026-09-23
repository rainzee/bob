from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import DemoSettings

import bub.configure as configure
from bub.builtin.settings import AgentSettings


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
        loaded = configure.load(config_file)

        assert loaded["model"] == "openai:gpt-5"
        assert loaded["demo"]["token"] == expected_token
        assert configure.ensure_config(AgentSettings).model == "openai:gpt-5"
        assert configure.ensure_config(DemoSettings).token == expected_token


def test_get_value_reads_registered_section_from_yaml(load_config) -> None:
    with patch.dict(os.environ, {}, clear=True):
        load_config(
            """
demo:
  token: yaml-token
""".strip(),
        )

        assert configure.get_value("demo.token") == "yaml-token"


def test_get_value_prefers_registered_env_over_yaml(load_config) -> None:
    load_config(
        """
demo:
  token: yaml-token
""".strip(),
    )

    with patch.dict(os.environ, {"BUB_DEMO_TOKEN": "env-token"}, clear=True):
        configure._global_config.clear()

        assert configure.get_value("demo.token") == "env-token"


def test_get_value_descends_into_registered_dict_field(load_config) -> None:
    with patch.dict(os.environ, {}, clear=True):
        load_config(
            """
api_key:
  openai: sk-yaml
""".strip(),
        )

        assert configure.get_value("api_key") == {"openai": "sk-yaml"}
        assert configure.get_value("api_key.openai") == "sk-yaml"


def test_get_value_ignores_raw_unregistered_path(load_config) -> None:
    load_config(
        """
custom:
  nested:
    value: raw-value
""".strip(),
    )

    with pytest.raises(KeyError):
        configure.get_value("custom.nested.value")


def test_get_value_returns_default_for_missing_path() -> None:
    assert configure.get_value("missing.value", default="fallback") == "fallback"
