from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ConfigDict, Field

from bub.builtin.hooks import Battery, BuiltinHooks
from bub.configure import Config, Settings, config
from bub.framework import BubFramework


@config("demo")
class DemoSettings(Settings):
    """Stand-in for a settings-owned config section"""

    model_config = ConfigDict(extra="ignore")

    token: str = Field(default="")


def install_builtin(framework: BubFramework, *, batteries: bool = False) -> Battery | None:
    """把 builtin 回调装到 framework 上, batteries=True 时一并装可选电池"""

    framework.add_hooks(BuiltinHooks(framework).hooks)
    if not batteries:
        return None
    battery = Battery(home=framework.home, config=framework.config)
    framework.add_hooks(battery.hooks)
    framework.add_tape_store(battery.tape_store)
    framework.add_sidecars(*battery.sidecars)
    framework.add_lifespans(*battery.lifespans)
    return battery


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
        return Config.from_file(config_file)

    return _load
