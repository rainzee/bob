"""AsyncStreamEvents 的关闭语义: 源被关闭, on_close 恰好一次"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from bub.errors import BubError, ErrorKind
from bub.streaming import AsyncStreamEvents, StreamEvent, StreamState


class _Source:
    """记录 aclose 与异常的假事件源"""

    def __init__(self, events: list[StreamEvent] | None = None, error: BaseException | None = None) -> None:
        self.events = events or []
        self.error = error
        self.closed = 0

    async def __aiter__(self):
        return self

    async def __anext__(self) -> StreamEvent:
        if self.events:
            return self.events.pop(0)
        if self.error is not None:
            raise self.error
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed += 1


class _Closer:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1


@pytest.mark.asyncio
async def test_exhaustion_closes_the_source_and_calls_on_close_once() -> None:
    source = _Source([StreamEvent("text", {"delta": "a"})])
    closer = _Closer()
    events = AsyncStreamEvents(source, on_close=closer)

    assert [event.data["delta"] async for event in events] == ["a"]

    assert source.closed == 1
    assert closer.calls == 1


@pytest.mark.asyncio
async def test_aclose_before_iterating_closes_the_source() -> None:
    source = _Source([StreamEvent("text", {"delta": "a"})])
    closer = _Closer()
    events = AsyncStreamEvents(source, on_close=closer)

    await events.aclose()

    assert source.closed == 1
    assert closer.calls == 1
    with pytest.raises(StopAsyncIteration):
        await events.__anext__()


@pytest.mark.asyncio
async def test_aclose_is_idempotent() -> None:
    source = _Source()
    closer = _Closer()
    events = AsyncStreamEvents(source, on_close=closer)

    await events.aclose()
    await events.aclose()
    assert [event async for event in events] == []

    assert source.closed == 1
    assert closer.calls == 1


@pytest.mark.asyncio
async def test_source_failure_closes_and_propagates() -> None:
    source = _Source(error=RuntimeError("stream broke"))
    closer = _Closer()
    events = AsyncStreamEvents(source, on_close=closer)

    with pytest.raises(RuntimeError, match="stream broke"):
        await events.__anext__()

    assert source.closed == 1
    assert closer.calls == 1


@pytest.mark.asyncio
async def test_closing_the_consumer_closes_the_inner_stream() -> None:
    """The outer stream owns the inner one, so closing the outer closes both."""

    async def inner() -> AsyncGenerator[StreamEvent, None]:
        yield StreamEvent("text", {"delta": "a"})
        yield StreamEvent("text", {"delta": "b"})

    source = AsyncStreamEvents(inner())
    events = AsyncStreamEvents(source)

    assert (await events.__anext__()).data["delta"] == "a"
    await events.aclose()

    with pytest.raises(StopAsyncIteration):
        await source.__anext__()


@pytest.mark.asyncio
async def test_error_and_usage_read_the_shared_state() -> None:
    state = StreamState()
    events = AsyncStreamEvents(_Source(), state=state)

    state.error = BubError(ErrorKind.TOOL, "boom")
    state.usage = {"total_tokens": 3}

    assert events.error is state.error
    assert events.usage == {"total_tokens": 3}


@pytest.mark.asyncio
async def test_state_defaults_to_a_fresh_stream_state() -> None:
    events = AsyncStreamEvents(_Source())

    assert events.error is None
    assert events.usage is None
