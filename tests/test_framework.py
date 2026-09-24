from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from bub.framework import BubFramework
from bub.hooks import Hooks


@pytest.mark.asyncio
async def test_hooks_are_appended_in_order_and_join_system_prompts(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)

    def low(prompt: str, state: dict[str, str]) -> str:
        return "low"

    async def high(prompt: str, state: dict[str, str]) -> str | None:
        return "high"

    def empty(prompt: str, state: dict[str, str]) -> str | None:
        return None

    framework.add_hooks(Hooks(system_prompt=[low]))
    framework.add_hooks(Hooks(system_prompt=[high, empty]))

    prompt = await framework.hooks.run_system_prompt("hello", {})
    assert prompt == "low\n\nhigh"


@pytest.mark.asyncio
async def test_add_hooks_keeps_sequence_order_across_calls(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)
    order: list[str] = []

    def first(session_id: str, state: dict) -> dict:
        order.append("first")
        return {"from": "first"}

    def second(session_id: str, state: dict) -> dict:
        order.append("second")
        return {"from": "second"}

    framework.add_hooks(Hooks(load_state=[first]))
    framework.add_hooks(Hooks(load_state=[second]))

    state = await framework.build_state("s")

    assert order == ["first", "second"]
    assert state["from"] == "second"


def test_add_sidecars_prefers_the_last_mount_for_a_duplicate_name(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)

    class Sidecar:
        def __init__(self, name: str, source: str) -> None:
            self.name = name
            self.source = source

    framework.add_sidecars(Sidecar("shared", "low"), Sidecar("low-only", "low"))
    framework.add_sidecars(Sidecar("shared", "high"), Sidecar("high-only", "high"))

    sidecars = {sidecar.name: sidecar for sidecar in framework.get_tape_sidecars()}

    assert set(sidecars) == {"shared", "low-only", "high-only"}
    assert cast(Any, sidecars["shared"]).source == "high"


def test_plugin_machinery_is_gone(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)

    assert not hasattr(framework, "plugin_manager")
    assert not hasattr(framework, "load_hooks")
    assert not hasattr(framework, "load_builtin_hooks")
    assert not hasattr(framework, "get_agent_hooks")
    assert not hasattr(framework, "continue_prompt")


@pytest.mark.asyncio
async def test_running_enters_tape_store_once_and_reuses_it(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)

    class RecordingTapeStore:
        def __init__(self) -> None:
            self.enter_count = 0
            self.exit_count = 0

    tape_store = RecordingTapeStore()

    def tape_store_lifespan():
        tape_store.enter_count += 1
        try:
            yield tape_store
        finally:
            tape_store.exit_count += 1

    framework.add_tape_store(tape_store_lifespan())

    async with framework.running():
        assert framework.get_tape_store() is tape_store
        assert framework.get_tape_store() is tape_store
        assert tape_store.enter_count == 1
        assert tape_store.exit_count == 0

    assert tape_store.enter_count == 1
    assert tape_store.exit_count == 1


@pytest.mark.asyncio
async def test_running_enters_every_registered_lifespan(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)
    entered: list[str] = []

    def first():
        entered.append("first-in")
        try:
            yield
        finally:
            entered.append("first-out")

    def second():
        entered.append("second-in")
        try:
            yield
        finally:
            entered.append("second-out")

    framework.add_lifespans(first, second)

    async with framework.running():
        assert entered == ["first-in", "second-in"]

    assert entered == ["first-in", "second-in", "second-out", "first-out"]


def test_framework_has_no_configuration_surface(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)

    assert not hasattr(framework, "config")
    assert not hasattr(framework, "settings")


@pytest.mark.asyncio
async def test_build_state_merges_defaults_seeds_and_load_state_callbacks(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)

    def low(session_id: str, state: dict[str, str]) -> dict[str, str]:
        assert state["_runtime_workspace"] == str(tmp_path)
        assert state["seed"] == "kept"
        return {"session_id": session_id, "from": "low"}

    def high(session_id: str, state: dict[str, str]) -> dict[str, str]:
        return {"from": "high"}

    framework.add_hooks(Hooks(load_state=[low, high]))

    state = await framework.build_state("session-1", {"seed": "kept"})

    assert state == {
        "_runtime_workspace": str(tmp_path),
        "seed": "kept",
        "session_id": "session-1",
        "from": "high",
    }


def test_process_inbound_is_gone(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)

    assert not hasattr(framework, "process_inbound")
    assert not hasattr(framework, "build_prompt")
    assert not hasattr(framework, "resolve_session")


def test_settings_machinery_is_gone() -> None:
    import importlib.util

    import bub

    assert importlib.util.find_spec("bub.configure") is None
    assert importlib.util.find_spec("bub.builtin.settings") is None
    assert "Config" not in bub.__all__
    assert "Settings" not in bub.__all__
