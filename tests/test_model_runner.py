"""模型调用边界: ChatRequest 的形状, 以及流式响应的解析"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from conftest import RecordingClient, call, reasoning_chunk, text_chunk, tool_calls_chunk, usage_chunk

from bub.context import default_tape_context
from bub.errors import BubError, ErrorKind
from bub.hooks import Hooks, LlmCallDecision
from bub.model_runner import (
    ChatRequest,
    ModelRunner,
    parse_tool_call,
    tool_invocation,
)
from bub.store import AsyncTapeStoreAdapter, FileTapeStore, InMemoryTapeStore
from bub.tape import Tape, TapeContext
from bub.tools import Tool, ToolExecutor


def _tape(tmp_path: Path | None = None) -> Tape:
    store = FileTapeStore(tmp_path) if tmp_path is not None else InMemoryTapeStore()
    return Tape(AsyncTapeStoreAdapter(store), default_tape_context()).scoped("test-tape")


async def _reload(tmp_path: Path) -> Tape:
    return Tape(AsyncTapeStoreAdapter(FileTapeStore(tmp_path)), default_tape_context()).scoped("test-tape")


@pytest.mark.asyncio
async def test_text_and_usage_reach_events_and_the_tape(tmp_path: Path) -> None:
    client = RecordingClient([text_chunk("do"), text_chunk("ne"), usage_chunk({"total_tokens": 5})])
    runner = ModelRunner(client=client)
    store = InMemoryTapeStore()
    tape = Tape(AsyncTapeStoreAdapter(store), default_tape_context()).scoped("test-tape")
    await tape.ensure_bootstrap_anchor()

    events = [event async for event in runner.run(tape=tape, model="m", tools=[], system_prompt=None, prompt="hello")]

    assert [(event.kind, event.data) for event in events] == [
        ("text", {"delta": "do"}),
        ("text", {"delta": "ne"}),
        ("usage", {"usage": {"total_tokens": 5}, "elapsed_seconds": events[2].data["elapsed_seconds"]}),
        ("final", {"ok": True, "text": "done"}),
    ]
    run_events = [entry for entry in store.read("test-tape") or [] if entry.payload.get("name") == "run"]
    assert run_events[0].payload["data"]["usage"] == {"total_tokens": 5}


@pytest.mark.asyncio
async def test_reasoning_delta_is_surfaced_without_becoming_text() -> None:
    client = RecordingClient([reasoning_chunk("thinking"), text_chunk("answer")])
    runner = ModelRunner(client=client)
    tape = _tape()
    await tape.ensure_bootstrap_anchor()

    events = [event async for event in runner.run(tape=tape, model="m", tools=[], system_prompt=None, prompt="hi")]

    assert [(event.kind, event.data) for event in events] == [
        ("reasoning", {"delta": "thinking"}),
        ("text", {"delta": "answer"}),
        ("usage", {"usage": None, "elapsed_seconds": events[2].data["elapsed_seconds"]}),
        ("final", {"ok": True, "text": "answer"}),
    ]


@pytest.mark.asyncio
async def test_streamed_tool_call_arguments_are_accumulated() -> None:
    client = RecordingClient([
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "ins", "arguments": '{"a"'}}]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "pect", "arguments": ": 1}"}}]}}]},
    ])
    runner = ModelRunner(client=client)
    tape = _tape()
    await tape.ensure_bootstrap_anchor()
    seen: list[dict[str, Any]] = []

    def inspect(a: int) -> str:
        seen.append({"a": a})
        return "ok"

    tools = [Tool.from_callable(inspect, name="inspect")]
    events = [event async for event in runner.run(tape=tape, model="m", tools=tools, system_prompt=None, prompt="hi")]

    (tool_call_event,) = [event for event in events if event.kind == "tool_call"]
    assert tool_call_event.data["tool_calls"] == [
        {"id": "call-1", "type": "function", "function": {"name": "inspect", "arguments": '{"a": 1}'}}
    ]
    assert seen == [{"a": 1}]


@pytest.mark.asyncio
async def test_unknown_tool_placeholder_surfaces_error_without_hooks() -> None:
    invocation = tool_invocation(
        {"id": "call-1", "type": "function", "function": {"name": "missing", "arguments": "{}"}}, {}
    )

    execution = await ToolExecutor().execute_async([invocation])

    assert execution.error is not None
    assert "missing" in execution.error.message


@pytest.mark.parametrize("arguments", ["[]", "null", "1", "not json"])
def test_parse_tool_call_rejects_non_object_arguments(arguments: str) -> None:
    with pytest.raises(BubError) as exc_info:
        parse_tool_call({"type": "function", "function": {"name": "inspect", "arguments": arguments}})

    assert exc_info.value.kind == ErrorKind.INVALID_INPUT


def test_parse_tool_call_rejects_a_non_function_call() -> None:
    with pytest.raises(BubError) as exc_info:
        parse_tool_call({"type": "custom", "custom": {"name": "inspect", "input": "raw"}})

    assert exc_info.value.kind == ErrorKind.INVALID_INPUT


def test_parse_tool_call_rejects_a_missing_name() -> None:
    with pytest.raises(BubError) as exc_info:
        parse_tool_call({"type": "function", "function": {"arguments": "{}"}})

    assert exc_info.value.kind == ErrorKind.INVALID_INPUT


@pytest.mark.asyncio
async def test_the_request_carries_messages_tools_and_the_resolved_model() -> None:
    client = RecordingClient()
    runner = ModelRunner(client=client)
    tape = _tape()
    await tape.ensure_bootstrap_anchor()
    tools = [Tool(name="inspect", handler=lambda: "found", description="look")]

    [
        event
        async for event in runner.run(tape=tape, model="openai:gpt-5", tools=tools, system_prompt="SYS", prompt="hi")
    ]

    (request,) = client.requests
    assert request.model == "openai:gpt-5"
    assert request.messages == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
    ]
    assert request.tools == [
        {"type": "function", "function": {"name": "inspect", "description": "look", "parameters": {}}}
    ]
    assert request.options == {}


@pytest.mark.asyncio
async def test_no_tools_sends_no_tool_payloads() -> None:
    client = RecordingClient()
    runner = ModelRunner(client=client)
    tape = _tape()
    await tape.ensure_bootstrap_anchor()

    [event async for event in runner.run(tape=tape, model="m", tools=[], system_prompt=None, prompt="hi")]

    assert client.requests[0].tools is None


@pytest.mark.asyncio
async def test_reasoning_effort_comes_from_tape_state() -> None:
    client = RecordingClient()
    runner = ModelRunner(client=client)
    tape = Tape(AsyncTapeStoreAdapter(InMemoryTapeStore()), TapeContext(state={"reasoning_effort": "high"})).scoped("t")
    await tape.ensure_bootstrap_anchor()

    [event async for event in runner.run(tape=tape, model="m", tools=[], system_prompt=None, prompt="hi")]

    assert client.requests[0].reasoning_effort == "high"


@pytest.mark.asyncio
async def test_options_and_max_tokens_reach_the_client() -> None:
    client = RecordingClient()
    runner = ModelRunner(client=client, max_tokens=42, options={"temperature": 0.2})
    tape = _tape()
    await tape.ensure_bootstrap_anchor()

    [event async for event in runner.run(tape=tape, model="m", tools=[], system_prompt=None, prompt="hi")]

    request = client.requests[0]
    assert request.max_tokens == 42
    assert request.options == {"temperature": 0.2}


@pytest.mark.asyncio
async def test_no_token_cap_is_sent_when_max_tokens_is_unset() -> None:
    client = RecordingClient()
    runner = ModelRunner(client=client)
    tape = _tape()
    await tape.ensure_bootstrap_anchor()

    [event async for event in runner.run(tape=tape, model="m", tools=[], system_prompt=None, prompt="hi")]

    assert client.requests[0].max_tokens is None


@pytest.mark.asyncio
async def test_before_llm_call_finish_skips_the_client() -> None:
    client = RecordingClient()
    hooks = Hooks(before_llm_call=[lambda request, state: LlmCallDecision.finish("stopped by policy")])
    runner = ModelRunner(client=client, hooks=hooks)
    tape = _tape()
    await tape.ensure_bootstrap_anchor()

    events = [event async for event in runner.run(tape=tape, model="m", tools=[], system_prompt=None, prompt="hi")]

    assert [event.data.get("delta") or event.data.get("text") for event in events] == [
        "stopped by policy",
        "stopped by policy",
    ]
    assert client.requests == []


@pytest.mark.asyncio
async def test_before_llm_call_can_rewrite_the_request() -> None:
    from dataclasses import replace

    def reroute(request, state):
        return replace(request, model="anthropic:new", max_tokens=7)

    client = RecordingClient()
    runner = ModelRunner(client=client, hooks=Hooks(before_llm_call=[reroute]))
    tape = _tape()
    await tape.ensure_bootstrap_anchor()

    [event async for event in runner.run(tape=tape, model="openai:orig", tools=[], system_prompt=None, prompt="hi")]

    assert client.requests[0].model == "anthropic:new"
    assert client.requests[0].max_tokens == 7


@pytest.mark.asyncio
async def test_the_client_stream_is_closed_when_the_consumer_finishes() -> None:
    client = RecordingClient([text_chunk("a"), text_chunk("b")])
    runner = ModelRunner(client=client)
    tape = _tape()
    await tape.ensure_bootstrap_anchor()

    [event async for event in runner.run(tape=tape, model="m", tools=[], system_prompt=None, prompt="hi")]

    assert client.closed == 1


@pytest.mark.asyncio
async def test_the_client_stream_is_closed_when_the_consumer_stops_early() -> None:
    async def endless() -> AsyncIterator[dict[str, Any]]:
        try:
            while True:
                yield text_chunk("x")
        finally:
            closed.append(True)

    closed: list[bool] = []

    class Client:
        def stream(self, request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
            return endless()

    runner = ModelRunner(client=Client())
    tape = _tape()
    await tape.ensure_bootstrap_anchor()

    stream = runner.run(tape=tape, model="m", tools=[], system_prompt=None, prompt="hi")
    assert (await stream.__anext__()).kind == "text"
    await stream.aclose()

    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [None, "", "Check byte equality.\nOnly exact equality counts."])
async def test_tool_call_text_survives_into_next_request_after_tape_reload(tmp_path: Path, content: str | None) -> None:
    calls = [call("inspect"), call("compare")]
    script = [text_chunk(part) for part in (content or "").splitlines(keepends=True)] + [tool_calls_chunk(calls)]
    client = RecordingClient(script, [text_chunk("done")])
    runner = ModelRunner(client=client)
    tools = [
        Tool(name="inspect", handler=lambda: "files found"),
        Tool(name="compare", handler=lambda: "bytes differ"),
    ]
    root = _tape(tmp_path)
    async with root.fork_tape() as tape:
        await tape.ensure_bootstrap_anchor()
        events = [
            event
            async for event in runner.run(
                tape=tape, model="m", tools=tools, system_prompt=None, prompt="Compare the outputs."
            )
        ]

    async for _ in runner.run(
        tape=await _reload(tmp_path), model="m", tools=tools, system_prompt=None, prompt="Continue."
    ):
        pass

    assert "".join(event.data["delta"] for event in events if event.kind == "text") == (content or "")
    assert client.requests[1].messages == [
        {"role": "user", "content": "Compare the outputs."},
        {"role": "assistant", "content": content or "", "tool_calls": calls},
        {"role": "tool", "content": "files found", "tool_call_id": "call-inspect", "name": "inspect"},
        {"role": "tool", "content": "bytes differ", "tool_call_id": "call-compare", "name": "compare"},
        {"role": "user", "content": "Continue."},
    ]


@pytest.mark.asyncio
async def test_multimodal_parts_survive_a_tool_call_and_a_tape_reload(tmp_path: Path) -> None:
    """A prompt the caller sent must still be there on the next step and after a restart.

    The tape is the conversation: anything recorded from the prompt has to come back
    out of it unchanged, or the model silently loses what it was asked about.
    """

    parts = [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "input_audio", "input_audio": {"data": "BBBB", "format": "wav"}},
    ]
    client = RecordingClient([tool_calls_chunk([call("noop")])], [text_chunk("done")])
    runner = ModelRunner(client=client)
    tools = [Tool(name="noop", handler=lambda: "done")]
    root = _tape(tmp_path)

    async with root.fork_tape() as tape:
        await tape.ensure_bootstrap_anchor()
        async for _ in runner.run(tape=tape, model="m", tools=tools, system_prompt=None, prompt=parts):
            pass

    async for _ in runner.run(
        tape=await _reload(tmp_path), model="m", tools=tools, system_prompt=None, prompt="Continue."
    ):
        pass

    for step, request in enumerate(client.requests, start=1):
        user_messages = [m for m in request.messages if m["role"] == "user" and isinstance(m["content"], list)]
        assert user_messages[0]["content"] == parts, f"step {step} lost part of the prompt"
