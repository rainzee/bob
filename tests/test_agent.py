from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from conftest import RecordingClient

from bub import BubFramework
from bub.agent import Agent
from bub.errors import BubError
from bub.hooks import Hooks
from bub.model_runner import ModelRunner
from bub.streaming import AsyncStreamEvents, StreamEvent, StreamState
from bub.tape import TapeContext
from bub.tools import tool

# ---------------------------------------------------------------------------
# Agent.run() tests: merge_back logic and model passthrough
# ---------------------------------------------------------------------------


class _FakeModelRunner(ModelRunner):
    def __init__(self, **options: Any) -> None:
        super().__init__(client=RecordingClient(), **options)
        self.run_calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> AsyncStreamEvents:
        self.run_calls.append(kwargs)

        async def events() -> AsyncIterator[StreamEvent]:
            yield StreamEvent("text", {"delta": "done"})
            yield StreamEvent("final", {"ok": True, "text": "done"})

        return AsyncStreamEvents(events())


def _make_agent() -> Agent:
    """Build an Agent with a mocked framework, bypassing real LLM/tape init."""
    framework = MagicMock()
    framework.get_tape_store.return_value = None
    framework.hooks = Hooks()
    framework.get_tape_sidecars.return_value = ()

    with patch.object(Agent, "__init__", lambda self, fw, **kwargs: None):
        agent = Agent.__new__(Agent)

    agent.framework = framework
    agent.model = "test:model"
    agent.tools = {}
    agent.tape_store = None
    agent.tape_context = TapeContext(state={})
    agent.sidecars = ()
    agent.hooks = Hooks()
    agent.max_steps = None
    agent.model_runner = _FakeModelRunner()
    return agent


def _model_runner(agent: Agent) -> _FakeModelRunner:
    assert isinstance(agent.model_runner, _FakeModelRunner)
    return agent.model_runner


class _ForkCapture:
    """Captures fork_tape enter and exit behavior."""

    def __init__(self) -> None:
        self.merge_back_values: list[bool] = []
        self.exit_count = 0

    @contextlib.asynccontextmanager
    async def fork_tape(self, tape_name: str, merge_back: bool = True) -> AsyncGenerator[None]:
        self.merge_back_values.append(merge_back)
        try:
            yield
        finally:
            self.exit_count += 1


class _FakeTape:
    """Scoped tape stand-in for testing Agent.run()."""

    def __init__(self, fork_capture: _ForkCapture) -> None:
        self._fork = fork_capture
        self.name = "test-tape"
        self.context = TapeContext(state={})
        self.messages: list[dict[str, Any]] = []
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    async def ensure_bootstrap_anchor(self) -> None:
        pass

    @contextlib.asynccontextmanager
    async def fork_tape(self, merge_back: bool = True) -> AsyncGenerator[_FakeTape]:
        async with self._fork.fork_tape(self.name, merge_back=merge_back):
            yield self

    async def read_messages(self) -> list[dict[str, Any]]:
        return list(self.messages)

    async def append_event(self, name: str, payload: dict[str, Any], **meta: Any) -> None:
        self.events.append((self.name, name, payload))

    async def record_chat(
        self,
        *,
        run_id: str,
        system_prompt: str | None,
        new_messages: list[dict[str, Any]],
        response_text: str | None,
        context_error: BubError | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[Any] | None = None,
        error: BubError | None = None,
        response: Any | None = None,
        provider: str | None = None,
        model: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        if system_prompt:
            self.events.append((self.name, "system", {"content": system_prompt}))
        if context_error is not None:
            self.events.append((self.name, "error", context_error.as_dict()))
        self.messages.extend(new_messages)
        if tool_calls:
            self.events.append((self.name, "tool_call", {"calls": tool_calls}))
        if tool_results is not None:
            self.events.append((self.name, "tool_result", {"results": tool_results}))
        if error is not None and error is not context_error:
            self.events.append((self.name, "error", error.as_dict()))
        if response_text is not None:
            self.messages.append({"role": "assistant", "content": response_text})
        self.events.append((self.name, "run", {"run_id": run_id, "model": model, "error": error is not None}))


class _FakeTapeFactory:
    """Minimal tape factory stand-in for testing Agent.run()."""

    def __init__(self, fork_capture: _ForkCapture) -> None:
        self.tape = _FakeTape(fork_capture)
        self.context = self.tape.context

    def session_tape(self, session_id: str, workspace: Any, context: TapeContext | None = None) -> _FakeTape:
        if context is not None:
            self.tape.context = context
            self.context = context
        return self.tape


@pytest.mark.asyncio
async def test_agent_run_regular_session_merges_back() -> None:
    """A regular (non-temp) session should merge tape entries back."""
    agent = _make_agent()
    fork_capture = _ForkCapture()
    agent.tape = _FakeTapeFactory(fork_capture)  # type: ignore[assignment]

    result = await agent.run_stream(session_id="user/session1", prompt="hello", state={"_runtime_workspace": "/tmp"})  # noqa: S108

    assert fork_capture.merge_back_values == [True]
    assert fork_capture.exit_count == 0

    [event async for event in result]

    assert fork_capture.merge_back_values == [True]
    assert fork_capture.exit_count == 1


@pytest.mark.asyncio
async def test_agent_run_temp_session_does_not_merge_back() -> None:
    """A temp/ session should NOT merge tape entries back."""
    agent = _make_agent()
    fork_capture = _ForkCapture()
    agent.tape = _FakeTapeFactory(fork_capture)  # type: ignore[assignment]

    result = await agent.run_stream(session_id="temp/abc123", prompt="hello", state={"_runtime_workspace": "/tmp"})  # noqa: S108

    assert fork_capture.merge_back_values == [False]
    assert fork_capture.exit_count == 0

    [event async for event in result]

    assert fork_capture.merge_back_values == [False]
    assert fork_capture.exit_count == 1


@pytest.mark.asyncio
async def test_agent_run_passes_model_to_llm() -> None:
    """The model parameter should be forwarded to the chat client."""
    agent = _make_agent()
    fork_capture = _ForkCapture()
    fake_tapes = _FakeTapeFactory(fork_capture)
    agent.tape = fake_tapes  # type: ignore[assignment]

    result = await agent.run_stream(
        session_id="user/s1",
        prompt="hello",
        state={"_runtime_workspace": "/tmp"},  # noqa: S108
        model="openai:gpt-4o",
    )
    [event async for event in result]

    (run_call,) = _model_runner(agent).run_calls
    assert run_call["model"] == "openai:gpt-4o"


@pytest.mark.asyncio
async def test_agent_run_empty_prompt_returns_error() -> None:
    agent = _make_agent()
    agent.tape = MagicMock()

    result = await agent.run_stream(session_id="user/s1", prompt="", state={})
    events = [event async for event in result]

    assert [(event.kind, event.data) for event in events] == [
        ("text", {"delta": "error: empty prompt"}),
        ("final", {"ok": False, "text": "error: empty prompt"}),
    ]


@pytest.mark.asyncio
async def test_agent_run_model_defaults_to_none() -> None:
    """When model is not specified, the client default is used."""
    agent = _make_agent()
    fork_capture = _ForkCapture()
    fake_tapes = _FakeTapeFactory(fork_capture)
    agent.tape = fake_tapes  # type: ignore[assignment]

    result = await agent.run_stream(session_id="user/s1", prompt="hello", state={"_runtime_workspace": "/tmp"})  # noqa: S108
    [event async for event in result]

    (run_call,) = _model_runner(agent).run_calls
    assert run_call["model"] == "test:model"


@pytest.mark.asyncio
async def test_agent_loop_continues_without_injecting_a_user_message() -> None:
    agent = _make_agent()
    tape = _FakeTape(_ForkCapture())
    prompts: list[str | list[dict] | None] = []
    prompt_texts: list[str] = []

    async def run_once(**kwargs: Any) -> AsyncStreamEvents:
        prompts.append(kwargs["prompt"])
        prompt_texts.append(kwargs["prompt_text"])
        should_continue = len(prompts) == 1

        async def iterator() -> AsyncIterator[StreamEvent]:
            yield StreamEvent("final", {"tool_calls": ["call"] if should_continue else []})

        return AsyncStreamEvents(iterator(), state=StreamState(usage={"step": len(prompts)}))

    agent._run_once = run_once  # type: ignore[method-assign]

    events = [
        event
        async for event in agent._stream_events(
            tape=tape,  # type: ignore[arg-type]
            prompt="initial prompt",
            state=StreamState(),
        )
    ]

    assert [event.kind for event in events] == ["final", "final"]
    # The first step carries the caller's message; the continuation asks for no new one.
    assert prompts == ["initial prompt", None]
    assert prompt_texts == ["initial prompt", "initial prompt"]


@pytest.mark.asyncio
async def test_agent_run_model_override_does_not_mutate_default() -> None:
    """A per-call model override must not leak into the agent's configured model.

    The override is resolved per turn (``model or self.model``) and
    forwarded to the client; it must never be written back to the agent model.
    This is the agent-layer half of the guarantee that a session-scoped model
    switch (state['model'] -> run_stream(model=...)) cannot bleed across
    sessions the way a process-global env var would.
    """
    agent = _make_agent()
    fork_capture = _ForkCapture()
    agent.tape = _FakeTapeFactory(fork_capture)  # type: ignore[assignment]
    default_model = agent.model

    result = await agent.run_stream(
        session_id="user/s1",
        prompt="hello",
        state={"_runtime_workspace": "/tmp"},  # noqa: S108
        model="openai:gpt-4o",
    )
    [event async for event in result]

    (run_call,) = _model_runner(agent).run_calls
    assert run_call["model"] == "openai:gpt-4o"
    assert agent.model == default_model


@pytest.mark.asyncio
async def test_agent_run_resolves_allowed_tool_aliases_and_limits_prompt() -> None:
    allowed_name = "tests.allowed_agent_tool"
    denied_name = "tests.denied_agent_tool"

    @tool(name=allowed_name, description="Allowed tool")
    def allowed_agent_tool() -> str:
        return "allowed"

    @tool(name=denied_name, description="Denied tool")
    def denied_agent_tool() -> str:
        return "denied"

    agent = _make_agent()
    agent.tools[allowed_name] = allowed_agent_tool
    agent.tools[denied_name] = denied_agent_tool
    fork_capture = _ForkCapture()
    fake_tapes = _FakeTapeFactory(fork_capture)
    agent.tape = fake_tapes  # type: ignore[assignment]

    result = await agent.run_stream(
        session_id="user/s1",
        prompt="hello",
        state={"_runtime_workspace": "/tmp"},  # noqa: S108
        allowed_tools=[" tests_allowed_agent_tool "],
    )
    [event async for event in result]

    (run_call,) = _model_runner(agent).run_calls
    assert [tool.name for tool in run_call["tools"]] == ["tests_allowed_agent_tool"]
    system_prompt = run_call["system_prompt"]
    assert "- tests_allowed_agent_tool(): Allowed tool" in system_prompt
    assert "tests_denied_agent_tool" not in system_prompt


@pytest.mark.asyncio
async def test_agent_run_rejects_unknown_allowed_tools() -> None:
    agent = _make_agent()
    fork_capture = _ForkCapture()
    fake_tapes = _FakeTapeFactory(fork_capture)
    agent.tape = fake_tapes  # type: ignore[assignment]

    stream = await agent.run_stream(
        session_id="user/s1",
        prompt="hello",
        state={"_runtime_workspace": "/tmp"},  # noqa: S108
        allowed_tools=["tests_missing_agent_tool"],
    )

    with pytest.raises(ValueError, match="tests_missing_agent_tool"):
        [event async for event in stream]


def test_agent_requires_an_explicit_model(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)

    with pytest.raises(TypeError, match="model"):
        Agent(framework, client=RecordingClient())


def test_agent_requires_a_client(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)

    with pytest.raises(TypeError, match="client"):
        Agent(framework, model="openai:test")


def test_agent_has_no_settings_or_config_object(tmp_path: Path) -> None:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)
    agent = Agent(framework, model="openai:test", client=RecordingClient())

    assert not hasattr(agent, "settings")
    assert not hasattr(framework, "config")
