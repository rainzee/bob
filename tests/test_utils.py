from pathlib import Path

import pytest

from bub.utils import workspace_from_state


def test_workspace_from_state_prefers_runtime_workspace_and_expands_user_home(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = Path.home().resolve()
    monkeypatch.setenv("HOME", str(expected))

    workspace = workspace_from_state({"_runtime_workspace": "~"})

    assert workspace == expected


def test_workspace_from_state_requires_an_explicit_workspace() -> None:
    with pytest.raises(ValueError, match="_runtime_workspace"):
        workspace_from_state({})

    with pytest.raises(ValueError, match="_runtime_workspace"):
        workspace_from_state({"_runtime_workspace": "   "})
