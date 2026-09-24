from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from bub.configure import Config, Settings, config


@config("demo")
class DemoSettings(Settings):
    """Stand-in for a plugin-owned config section"""

    model_config = SettingsConfigDict(env_prefix="BUB_DEMO_", extra="ignore")

    token: str = Field(default="")


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[[str], Path]:
    def _write(content: str = "") -> Path:
        config_file = tmp_path / "config.yml"
        config_file.write_text(content, encoding="utf-8")
        return config_file

    return _write


@pytest.fixture
def load_config(write_config: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch) -> Callable[[str], Config]:
    def _load(content: str = "") -> Config:
        config_file = write_config(content)
        monkeypatch.chdir(config_file.parent)
        loaded = Config()
        loaded.load(config_file)
        return loaded

    return _load
