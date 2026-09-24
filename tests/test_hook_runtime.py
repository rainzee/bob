import pluggy
import pytest

from bub.hooks import BUB_HOOK_NAMESPACE, BubHookSpecs, hookimpl
from bub.hooks.runtime import HookRuntime
from bub.turn import TurnState


def _runtime_with_plugins(*plugins: tuple[str, object]) -> HookRuntime:
    manager = pluggy.PluginManager(BUB_HOOK_NAMESPACE)
    manager.add_hookspecs(BubHookSpecs)
    for name, plugin in plugins:
        manager.register(plugin, name=name)
    return HookRuntime(manager)


@pytest.mark.asyncio
async def test_call_first_respects_priority_and_returns_first_non_none() -> None:
    called: list[str] = []

    class LowPriority:
        @hookimpl
        def continue_prompt(self, prompt, tape, state):
            called.append("low")
            return "low"

    class MidPriority:
        @hookimpl
        def continue_prompt(self, prompt, tape, state):
            called.append("mid")
            return "mid"

    class HighPriorityReturnsNone:
        @hookimpl
        def continue_prompt(self, prompt, tape, state):
            called.append("high")
            return None

    runtime = _runtime_with_plugins(
        ("low", LowPriority()),
        ("mid", MidPriority()),
        ("high", HighPriorityReturnsNone()),
    )

    result = await runtime.call_first("continue_prompt", prompt="p", tape=None, state=None, ignored="value")
    assert result == "mid"
    assert called == ["high", "mid"]


@pytest.mark.asyncio
async def test_call_many_collects_every_result_in_priority_order() -> None:
    seen: list[TurnState] = []

    class First:
        @hookimpl
        def load_state(self, session_id, state):
            seen.append(dict(state))
            return {"first": True}

    class Second:
        @hookimpl
        def load_state(self, session_id, state):
            seen.append(dict(state))
            return {"second": True}

    runtime = _runtime_with_plugins(("first", First()), ("second", Second()))

    results = await runtime.call_many("load_state", session_id="s", state={"seed": 1})

    assert results == [{"second": True}, {"first": True}]
    assert seen == [{"seed": 1}, {"seed": 1}]


def test_call_many_sync_skips_async_impl() -> None:
    class _AwaitableValue:
        def __await__(self):
            yield from ()
            return "async"

    class AsyncPrompt:
        @hookimpl
        def system_prompt(self, prompt, state):
            return _AwaitableValue()

    class SyncPrompt:
        @hookimpl
        def system_prompt(self, prompt, state):
            return "sync"

    runtime = _runtime_with_plugins(
        ("sync", SyncPrompt()),
        ("async", AsyncPrompt()),
    )

    assert runtime.call_many_sync("system_prompt", prompt="hello", state={}) == ["sync"]


@pytest.mark.asyncio
async def test_call_first_swallows_implementation_failures() -> None:
    class RaisingHook:
        @hookimpl
        def continue_prompt(self, prompt, tape, state):
            raise RuntimeError("boom")

    class WorkingHook:
        @hookimpl
        def continue_prompt(self, prompt, tape, state):
            return "ok"

    runtime = _runtime_with_plugins(("raise", RaisingHook()), ("working", WorkingHook()))

    assert await runtime.call_first("continue_prompt", prompt="p", tape=None, state=None) == "ok"


def test_removed_hooks_are_not_registered() -> None:
    spec_names = {name for name in dir(BubHookSpecs) if not name.startswith("_")}

    assert spec_names == {
        "after_llm_call",
        "after_tool_call",
        "before_llm_call",
        "before_tool_call",
        "build_tape_context",
        "continue_prompt",
        "load_state",
        "provide_lifespan",
        "provide_tape_sidecar",
        "provide_tape_store",
        "system_prompt",
    }
