from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bub.builtin import battery_tools
from bub.builtin.hook_impl import (
    AGENTS_FILE_NAME,
    DEFAULT_SYSTEM_PROMPT,
    BatteryImpl,
    BuiltinImpl,
)
from bub.configure import Config
from bub.framework import BubFramework
from bub.store import AsyncTapeStoreAdapter, FileTapeStore, InMemoryTapeStore
from bub.streaming import AsyncStreamEvents, StreamEvent
from bub.tape import Tape, TapeContext


def _fake_tape(home: Path) -> Tape:
    return Tape(
        archive_path=home / "tapes",
        store=AsyncTapeStoreAdapter(InMemoryTapeStore()),
        context=TapeContext(),
    )


class FakeAgent:
    def __init__(self, home: Path, *, tape: Tape | None = None) -> None:
        self.settings = SimpleNamespace(home=home)
        self.tools = {tool_item.name: tool_item for tool_item in battery_tools()}
        # A real in-memory async tape so load_state's recovery path runs against
        # the same store the tests write `model_switch` events to.
        self.tape = tape if tape is not None else _fake_tape(home)
        self.run_calls: list[tuple[str, str, dict[str, object]]] = []
        self.run_stream_calls: list[tuple[str, str, dict[str, object], str | None]] = []

    async def run(self, *, session_id: str, prompt: str, state: dict[str, object]) -> str:
        self.run_calls.append((session_id, prompt, state))
        return "agent-output"

    async def run_stream(
        self,
        *,
        session_id: str,
        prompt: str,
        state: dict[str, object],
        model: str | None = None,
    ) -> AsyncStreamEvents:
        self.run_stream_calls.append((session_id, prompt, state, model))

        async def iterator():
            yield StreamEvent("text", {"delta": "agent-output"})

        return AsyncStreamEvents(iterator())


def _raise_value_error() -> None:
    raise ValueError("boom")


def _build_impl(tmp_path: Path, config: Config | None = None) -> tuple[BubFramework, BuiltinImpl, FakeAgent]:
    framework = BubFramework(
        workspace=tmp_path,
        home=tmp_path,
        config=config if config is not None else Config({"model": "test:model"}),
    )
    impl = BuiltinImpl(framework)
    agent = FakeAgent(tmp_path)
    impl._agent = agent
    return framework, impl, agent


def test_resolve_session_is_gone(tmp_path: Path) -> None:
    _, impl, _ = _build_impl(tmp_path)

    assert not hasattr(impl, "resolve_session")
    assert not hasattr(impl, "continue_prompt")


@pytest.mark.asyncio
async def test_load_state_uses_the_seeded_agent_and_defaults(tmp_path: Path) -> None:
    _, impl, agent = _build_impl(tmp_path)

    state = await impl.load_state(session_id="resolved-session", state={"_runtime_agent": agent})

    assert state["session_id"] == "resolved-session"
    assert state["_runtime_agent"] is agent


@pytest.mark.asyncio
async def test_load_state_injects_model_recorded_on_session_tape(tmp_path: Path) -> None:
    """A model_switch event recorded on the session tape is restored into state on load."""
    _, impl, agent = _build_impl(tmp_path)
    session = agent.tape.session_tape("resolved-session", impl.framework.workspace)
    await session.append_event("model_switch", {"model": "openai:gpt-4o"})

    state = await impl.load_state(session_id="resolved-session", state={})

    assert state["model"] == "openai:gpt-4o"


@pytest.mark.asyncio
async def test_load_state_injects_reasoning_effort_recorded_on_session_tape(tmp_path: Path) -> None:
    _, impl, agent = _build_impl(tmp_path)
    session = agent.tape.session_tape("resolved-session", impl.framework.workspace)
    await session.append_event("reasoning_effort_switch", {"reasoning_effort": "high"})

    state = await impl.load_state(session_id="resolved-session", state={})

    assert state["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_load_state_does_not_inject_model_for_unknown_session(tmp_path: Path) -> None:
    """A session with nothing recorded on its tape must not inherit any model (no leakage)."""
    _, impl, _ = _build_impl(tmp_path)

    state = await impl.load_state(session_id="fresh-session", state={})

    assert "model" not in state


@pytest.mark.asyncio
async def test_recover_session_model_returns_latest_recorded(tmp_path: Path) -> None:
    """When several switches were recorded, the most recent one wins."""
    _, impl, agent = _build_impl(tmp_path)
    session = agent.tape.session_tape("resolved-session", impl.framework.workspace)
    await session.append_event("model_switch", {"model": "openai:gpt-4o"})
    await session.append_event("model_switch", {"model": "anthropic:claude-3"})

    assert await impl._recover_session_model("resolved-session", agent=agent) == "anthropic:claude-3"


def test_system_prompt_appends_workspace_agents_file(tmp_path: Path) -> None:
    _, impl, _ = _build_impl(tmp_path)
    (tmp_path / AGENTS_FILE_NAME).write_text("local rules", encoding="utf-8")

    result = impl.system_prompt(prompt="hello", state={"_runtime_workspace": str(tmp_path)})

    assert result == DEFAULT_SYSTEM_PROMPT + "\n\nlocal rules"


def test_system_prompt_ignores_missing_agents_file(tmp_path: Path) -> None:
    _, impl, _ = _build_impl(tmp_path)

    result = impl.system_prompt(prompt="hello", state={"_runtime_workspace": str(tmp_path)})

    assert result == DEFAULT_SYSTEM_PROMPT + "\n\n"


def test_battery_impl_provides_file_tape_store(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)

    store = BatteryImpl(framework).provide_tape_store()

    assert isinstance(store, FileTapeStore)
    assert store._directory == tmp_path / "tapes"


def test_before_tool_call_ignores_known_tool(tmp_path: Path) -> None:
    _, impl, _ = _build_impl(tmp_path)
    import asyncio

    from bub.hooks.interception import ToolCall

    async def _do():
        return await impl.before_tool_call(ToolCall(run_id="r", tool="bash", arguments={}), state={})

    assert asyncio.run(_do()) is None


def test_before_tool_call_ignores_known_model_alias(tmp_path: Path) -> None:
    _, impl, _ = _build_impl(tmp_path)
    import asyncio

    from bub.hooks.interception import ToolCall

    async def _do():
        return await impl.before_tool_call(ToolCall(run_id="r", tool="bash_output", arguments={}), state={})

    assert asyncio.run(_do()) is None


def test_before_tool_call_recovers_unknown_tool(tmp_path: Path) -> None:
    _, impl, _ = _build_impl(tmp_path)
    import asyncio

    from bub.hooks.interception import ToolCall

    async def _do():
        return await impl.before_tool_call(ToolCall(run_id="r", tool="tepadr", arguments={}), state={})

    decision = asyncio.run(_do())
    assert decision is not None and decision.action == "replace"
    assert "tepadr" in decision.result
    assert "skill" in decision.result


def test_before_tool_call_suggests_close_model_tool_name(tmp_path: Path) -> None:
    _, impl, _ = _build_impl(tmp_path)
    import asyncio

    from bub.hooks.interception import ToolCall

    async def _do():
        return await impl.before_tool_call(ToolCall(run_id="r", tool="fs_reed", arguments={}), state={})

    decision = asyncio.run(_do())
    assert decision is not None
    assert "fs_reed" in decision.result
    assert "fs_read" in decision.result
