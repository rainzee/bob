from __future__ import annotations

import importlib.metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from conftest import DemoSettings

from bub.builtin.settings import AgentSettings
from bub.configure import Config
from bub.framework import BubFramework
from bub.hooks import hookimpl
from bub.streaming import StreamState


def test_get_system_prompt_uses_priority_order_and_skips_empty_results(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)

    class LowPriorityPlugin:
        @hookimpl
        def system_prompt(self, prompt: str, state: dict[str, str]) -> str:
            return "low"

    class HighPriorityPlugin:
        @hookimpl
        def system_prompt(self, prompt: str, state: dict[str, str]) -> str | None:
            return "high"

    class EmptyPlugin:
        @hookimpl
        def system_prompt(self, prompt: str, state: dict[str, str]) -> str | None:
            return None

    framework.plugin_manager.register(LowPriorityPlugin(), name="low")
    framework.plugin_manager.register(HighPriorityPlugin(), name="high")
    framework.plugin_manager.register(EmptyPlugin(), name="empty")

    prompt = framework.get_system_prompt(prompt="hello", state={})

    assert prompt == "low\n\nhigh"


def test_get_tape_sidecars_combines_plugins_and_prefers_the_highest_priority_name(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)

    class Sidecar:
        def __init__(self, name: str, source: str) -> None:
            self.name = name
            self.source = source

    class SidecarPlugin:
        def __init__(self, sidecar: Sidecar) -> None:
            self.sidecar = sidecar

        @hookimpl
        def provide_tape_sidecar(self) -> Sidecar:
            return self.sidecar

    framework.plugin_manager.register(SidecarPlugin(Sidecar("shared", "low")), name="low-shared")
    framework.plugin_manager.register(SidecarPlugin(Sidecar("low-only", "low")), name="low-only")
    framework.plugin_manager.register(SidecarPlugin(Sidecar("shared", "high")), name="high-shared")
    framework.plugin_manager.register(SidecarPlugin(Sidecar("high-only", "high")), name="high-only")

    sidecars = {sidecar.name: sidecar for sidecar in framework.get_tape_sidecars()}

    assert set(sidecars) == {"shared", "low-only", "high-only"}
    assert cast(Any, sidecars["shared"]).source == "high"


@pytest.mark.asyncio
async def test_continue_prompt_awaits_high_priority_async_hook(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)
    tape = cast(Any, SimpleNamespace(context=SimpleNamespace(state={})))
    state = StreamState(usage={"total_tokens": 42})
    called: list[str] = []

    class SyncPlugin:
        @hookimpl
        def continue_prompt(self, prompt, tape, state):
            called.append("sync")
            return "sync prompt"

    class AsyncPlugin:
        @hookimpl
        async def continue_prompt(self, prompt: str, tape: Any, state: StreamState) -> str:
            called.append("async")
            assert prompt == "current prompt"
            assert state.usage == {"total_tokens": 42}
            return "async prompt"

    framework.plugin_manager.register(SyncPlugin(), name="sync")
    framework.plugin_manager.register(AsyncPlugin(), name="async")

    prompt = await framework.continue_prompt(prompt="current prompt", tape=tape, state=state)

    assert prompt == "async prompt"
    assert called == ["async"]


@pytest.mark.asyncio
async def test_running_enters_tape_store_once_and_reuses_it(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)

    class RecordingTapeStore:
        def __init__(self) -> None:
            self.enter_count = 0
            self.exit_count = 0

    tape_store = RecordingTapeStore()

    class TapePlugin:
        @hookimpl
        def provide_tape_store(self):
            tape_store.enter_count += 1
            try:
                yield tape_store
            finally:
                tape_store.exit_count += 1

    framework.plugin_manager.register(TapePlugin(), name="tape")

    async with framework.running():
        assert framework.get_tape_store() is tape_store
        assert framework.get_tape_store() is tape_store
        assert tape_store.enter_count == 1
        assert tape_store.exit_count == 0

    assert tape_store.enter_count == 1
    assert tape_store.exit_count == 1


def test_load_hooks_loads_root_and_named_config_sections(write_config) -> None:
    expected = "test-token"
    config_file = write_config(
        f"""
model: openai:gpt-5
demo:
    token: {expected}
""".strip()
    )
    framework = BubFramework(
        workspace=config_file.parent, home=config_file.parent, config=Config.from_file(config_file)
    )

    framework.load_hooks()

    assert framework.config.ensure(AgentSettings).model == "openai:gpt-5"
    assert framework.config.ensure(DemoSettings).token == expected


def test_load_hooks_initializes_callable_plugins_after_config_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path, config=Config({"model": "openai:gpt-5"}))

    class SettingsAwarePlugin:
        def __init__(self, framework: BubFramework) -> None:
            self.model = framework.config.ensure(AgentSettings).model

        @hookimpl
        def provide_tape_store(self) -> None:
            return None

    entry_point = SimpleNamespace(name="config-plugin", load=lambda: SettingsAwarePlugin)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [entry_point])

    framework.load_hooks()

    plugin = framework.plugin_manager.get_plugin("config-plugin")
    assert isinstance(plugin, SettingsAwarePlugin)
    assert plugin.model == "openai:gpt-5"


@pytest.mark.asyncio
async def test_build_state_merges_defaults_seeds_and_load_state_hooks(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)

    class LowPriority:
        @hookimpl
        def load_state(self, session_id: str, state: dict[str, str]) -> dict[str, str]:
            assert state["_runtime_workspace"] == str(tmp_path)
            assert state["seed"] == "kept"
            return {"session_id": session_id, "from": "low"}

    class HighPriority:
        @hookimpl
        def load_state(self, session_id: str, state: dict[str, str]) -> dict[str, str]:
            return {"from": "high"}

    framework.plugin_manager.register(LowPriority(), name="low")
    framework.plugin_manager.register(HighPriority(), name="high")

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
