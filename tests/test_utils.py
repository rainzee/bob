from pathlib import Path

import pytest

from bub.utils import workspace_from_state


def test_workspace_from_state_prefers_runtime_workspace_and_expands_user_home(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = Path.home().resolve()
    monkeypatch.setenv("HOME", str(expected))

    workspace = workspace_from_state({"_runtime_workspace": "~"})

    assert workspace == expected


def test_workspace_from_state_falls_back_to_current_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    workspace = workspace_from_state({"_runtime_workspace": "   "})

    assert workspace == tmp_path.resolve()
