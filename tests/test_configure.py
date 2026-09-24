from __future__ import annotations

from pathlib import Path

import pytest
from conftest import DemoSettings

from bub.builtin.settings import AgentSettings
from bub.configure import Config

MODEL = "test:model"


def test_from_file_reads_registered_sections(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yml"
    expected_token = "123:abc"  # noqa: S105
    config_file.write_text(
        f"""
model: {MODEL}
demo:
  token: {expected_token}
""".strip(),
        encoding="utf-8",
    )

    config = Config.from_file(config_file)

    assert config.data == {"model": MODEL, "demo": {"token": expected_token}}
    assert config.ensure(AgentSettings).model == MODEL
    assert config.ensure(DemoSettings).token == expected_token


def test_from_file_treats_a_missing_file_as_empty_configuration(tmp_path: Path) -> None:
    config = Config.from_file(tmp_path / "config.yml")

    assert config.data == {}


def test_explicit_data_replaces_any_environment_reading() -> None:
    config = Config({"model": MODEL})

    assert config.ensure(AgentSettings).model == MODEL


def test_ensure_caches_within_one_config_and_not_across_instances() -> None:
    config = Config()

    assert config.ensure(DemoSettings) is config.ensure(DemoSettings)
    assert Config().ensure(DemoSettings) is not config.ensure(DemoSettings)


def test_get_value_reads_registered_section_from_data() -> None:
    config = Config({"demo": {"token": "yaml-token"}})

    assert config.get_value("demo.token") == "yaml-token"


def test_get_value_descends_into_registered_dict_field() -> None:
    config = Config({"model": MODEL, "api_key": {"openai": "sk-yaml"}})

    assert config.get_value("api_key") == {"openai": "sk-yaml"}
    assert config.get_value("api_key.openai") == "sk-yaml"


def test_get_value_ignores_raw_unregistered_path() -> None:
    config = Config({"model": MODEL, "custom": {"nested": {"value": "raw-value"}}})

    with pytest.raises(KeyError):
        config.get_value("custom.nested.value")


def test_get_value_returns_default_for_missing_path() -> None:
    assert Config({"model": MODEL}).get_value("missing.value", default="fallback") == "fallback"
