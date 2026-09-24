"""The model-call boundary, and the parsing of a streamed response into events"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Mapping
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import monotonic
from typing import Any, Protocol

from pydantic import TypeAdapter, ValidationError

from bub.errors import BubError, ErrorKind
from bub.hooks import Hooks, LlmCallDecision, LlmCallRequest, LlmCallResult
from bub.streaming import AsyncStreamEvents, StreamEvent, StreamState
from bub.tape import Tape
from bub.tools import Tool, ToolContext, ToolExecutor

TOOL_ARGUMENTS_ADAPTER = TypeAdapter(dict[str, Any])


@dataclass(frozen=True)
class ChatRequest:
    """One chat call handed to the host's client

    Messages and tools use the OpenAI chat-completions shape, which is also the shape
    the tape records and ``Tool.to_schema`` produces
    """

    run_id: str
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


class ChatClient(Protocol):
    """The host's own model call, implemented with whichever SDK it prefers

    Yield OpenAI chat-completion chunks as JSON mappings: each chunk carries
    ``choices[i].delta``, and the last one may carry ``usage``. A ``reasoning``
    delta is surfaced as a reasoning event; a provider that names it otherwise
    should rename it in its adapter.
    """

    def stream(self, request: ChatRequest) -> AsyncIterator[dict[str, Any]]: ...


class ModelRunner:
    def __init__(
        self,
        *,
        client: ChatClient,
        max_tokens: int | None = None,
        options: Mapping[str, Any] | None = None,
        timeout_seconds: float | None = None,
        hooks: Hooks | None = None,
    ) -> None:
        """Create a runner that turns one turn step into one chat call.

        Args:
            client: The host's chat client.
            max_tokens: Per-call output cap; None leaves it to the client.
            options: Extra options added to every request for the client to interpret.
            timeout_seconds: Per-call timeout; None means no timeout is applied.
            hooks: Callbacks run around each model call.
        """
        self.client = client
        self.max_tokens = max_tokens
        self.options = dict(options or {})
        self.timeout_seconds = timeout_seconds
        self.hooks = hooks

    def run(
        self,
        *,
        tape: Tape,
        model: str,
        tools: list[Tool],
        system_prompt: str | None,
        prompt: str | list[dict] | None,
    ) -> AsyncStreamEvents:
        state = StreamState()

        async def iterator() -> AsyncGenerator[StreamEvent]:
            run_id = self.generate_run_id()
            messages, new_messages = await self.build_messages(
                tape=tape,
                run_id=run_id,
                system_prompt=system_prompt,
                prompt=prompt,
                model=model,
            )
            output = ModelOutputAccumulator()
            request = LlmCallRequest(
                run_id=run_id,
                model=model,
                messages=messages,
                tool_names=tuple(tool_item.name for tool_item in tools),
                max_tokens=self.max_tokens,
            )
            decision: LlmCallDecision | None = None
            if self.hooks is not None:
                request, decision = await self.hooks.run_before_llm_call(request, tape.context.state)
            if decision is not None:
                await self.record_chat(
                    tape=tape,
                    run_id=run_id,
                    system_prompt=system_prompt,
                    new_messages=new_messages,
                    response_text=decision.text,
                    model=request.model,
                )
                yield StreamEvent("text", {"delta": decision.text})
                yield StreamEvent("final", {"ok": True, "text": decision.text})
                return
            llm_started = datetime.now(UTC)
            after_fired = False

            async def fire_after(error: Exception | None = None) -> None:
                """Fire after_llm_call once per completed call (success or Exception failure); cancellation/consumer close bypasses it."""

                nonlocal after_fired
                if after_fired:
                    return
                after_fired = True
                await self._fire_after_llm_call(request, output, state, llm_started, tape, error=error)

            try:
                completion_started = monotonic()
                async with aclosing(self._stream_completion(request, tools, tape, state, output)) as events:
                    async for event in events:
                        yield event
                completion_elapsed = monotonic() - completion_started
            except Exception as exc:
                # Cancellation / consumer close (BaseException) intentionally
                # bypasses after_llm_call: only real completions and failures
                # are terminal observations.
                await fire_after(exc)
                raise
            await fire_after()

            yield StreamEvent("usage", {"usage": state.usage, "elapsed_seconds": completion_elapsed})

            tool_calls = output.tool_calls
            if tool_calls:
                tool_map = {tool_item.name: tool_item for tool_item in tools}
                tool_invocations = [tool_invocation(tool_call, tool_map) for tool_call in tool_calls]
                yield StreamEvent("tool_call", {"tool_calls": tool_calls})
                context = ToolContext(tape=tape, run_id=run_id, state=tape.context.state)
                execution = await ToolExecutor(hooks=self.hooks).execute_async(tool_invocations, context=context)
                await self.record_chat(
                    tape=tape,
                    run_id=run_id,
                    system_prompt=system_prompt,
                    new_messages=new_messages,
                    response_text=output.text or None,
                    tool_calls=tool_calls,
                    tool_results=execution.tool_results,
                    model=request.model,
                    usage=state.usage,
                )
                yield StreamEvent("tool_result", {"tool_results": execution.tool_results})
                yield StreamEvent(
                    "final", {"ok": True, "tool_calls": tool_calls, "tool_results": execution.tool_results}
                )
                return

            text = output.text
            await self.record_chat(
                tape=tape,
                run_id=run_id,
                system_prompt=system_prompt,
                new_messages=new_messages,
                response_text=text,
                model=request.model,
                usage=state.usage,
            )
            yield StreamEvent("final", {"ok": True, "text": text})

        return AsyncStreamEvents(iterator(), state=state)

    async def _stream_completion(
        self,
        request: LlmCallRequest,
        tools: list[Tool],
        tape: Tape,
        state: StreamState,
        output: ModelOutputAccumulator,
    ) -> AsyncGenerator[StreamEvent]:
        chat_request = ChatRequest(
            run_id=request.run_id,
            model=request.model,
            messages=list(request.messages),
            tools=[tool.to_schema() for tool in tools] or None,
            max_tokens=request.max_tokens,
            reasoning_effort=tape.context.state.get("reasoning_effort"),
            options=self.options,
        )
        chunks = self.client.stream(chat_request)
        try:
            async with asyncio.timeout(self.timeout_seconds):
                async for chunk in chunks:
                    for event in self._chunk_events(chunk, state, output):
                        yield event
        finally:
            if close := getattr(chunks, "aclose", None):
                await close()

    def _chunk_events(
        self,
        chunk: Mapping[str, Any],
        state: StreamState,
        output: ModelOutputAccumulator,
    ) -> Iterable[StreamEvent]:
        if usage := chunk.get("usage"):
            state.usage = dict(usage)
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if reasoning := delta.get("reasoning"):
                yield StreamEvent("reasoning", {"delta": reasoning_text(reasoning)})
            if content := delta.get("content"):
                output.add_text(content)
                yield StreamEvent("text", {"delta": content})
            if tool_calls := delta.get("tool_calls"):
                output.merge_delta_tool_calls(tool_calls)

    @staticmethod
    def generate_run_id() -> str:
        return f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"

    async def _fire_after_llm_call(
        self,
        request: LlmCallRequest,
        output: ModelOutputAccumulator,
        state: StreamState,
        started: datetime,
        tape: Tape,
        error: Exception | None = None,
    ) -> None:
        if self.hooks is None:
            return
        duration_ms = int((datetime.now(UTC) - started).total_seconds() * 1000)
        result = LlmCallResult(
            run_id=request.run_id,
            text=output.text or None,
            tool_calls=output.tool_calls,
            usage=state.usage,
            error=error,
            duration_ms=duration_ms,
        )
        await self.hooks.run_after_llm_call(request, result, tape.context.state)

    async def build_messages(
        self,
        *,
        tape: Tape,
        run_id: str,
        system_prompt: str | None,
        prompt: str | list[dict] | None,
        model: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        try:
            messages = await tape.read_messages()
        except BubError as exc:
            await self.record_context_error(
                tape=tape,
                run_id=run_id,
                system_prompt=system_prompt,
                error=exc,
                model=model,
            )
            raise
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, *messages]
        # ``prompt is None`` continues an agent loop: the tape already ends with the
        # assistant tool calls and their results, so no new user message is added.
        new_messages: list[dict[str, Any]] = [] if prompt is None else [{"role": "user", "content": prompt}]
        messages.extend(new_messages)
        return messages, new_messages

    async def record_context_error(
        self,
        *,
        tape: Tape,
        run_id: str,
        system_prompt: str | None,
        error: BubError,
        model: str,
    ) -> None:
        await self.record_chat(
            tape=tape,
            run_id=run_id,
            system_prompt=system_prompt,
            context_error=error,
            new_messages=[],
            response_text=None,
            error=error,
            model=model,
        )

    async def record_chat(
        self,
        *,
        tape: Tape,
        run_id: str,
        system_prompt: str | None,
        new_messages: list[dict[str, Any]],
        response_text: str | None,
        context_error: BubError | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[Any] | None = None,
        error: BubError | None = None,
        model: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        await tape.record_chat(
            run_id=run_id,
            system_prompt=system_prompt,
            new_messages=new_messages,
            response_text=response_text,
            context_error=context_error,
            tool_calls=tool_calls,
            tool_results=tool_results,
            error=error,
            model=model,
            usage=usage,
        )


def reasoning_text(reasoning: object) -> str:
    content = reasoning.get("content") if isinstance(reasoning, Mapping) else reasoning
    return "" if content is None else str(content)


@dataclass
class StreamToolCall:
    id: str | None = None
    name: str | None = None
    arguments: str = ""

    def merge(self, delta: Mapping[str, Any]) -> None:
        if delta.get("id"):
            self.id = str(delta["id"])
        function = delta.get("function") or {}
        name = function.get("name")
        if name:
            # Some providers split the function name across deltas.
            self.name = name if self.name is None or self.name == name else self.name + name
        arguments = function.get("arguments")
        if arguments:
            self.arguments += arguments

    def as_message(self, index: int) -> dict[str, Any]:
        return {
            "id": self.id or f"call_{index}",
            "type": "function",
            "function": {"name": self.name or "", "arguments": self.arguments or "{}"},
        }


class ModelOutputAccumulator:
    def __init__(self) -> None:
        self._text_parts: list[str] = []
        self._stream_calls: dict[int, StreamToolCall] = {}

    def add_text(self, text: str) -> None:
        self._text_parts.append(text)

    def merge_delta_tool_calls(self, deltas: Iterable[Mapping[str, Any]]) -> None:
        for position, delta in enumerate(deltas):
            index = delta.get("index")
            self._stream_calls.setdefault(index if index is not None else position, StreamToolCall()).merge(delta)

    @property
    def text(self) -> str:
        return "".join(self._text_parts)

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return [call.as_message(index) for index, call in sorted(self._stream_calls.items())]


def tool_invocation(
    tool_call: Mapping[str, Any],
    tool_map: dict[str, Tool],
) -> tuple[Tool, dict[str, Any]]:
    """Resolve a model tool call to (runtime tool, arguments).

    An unknown tool name is not treated as a fatal error: it is surfaced as a
    placeholder ``Tool`` so the invocation flows through ``ToolExecutor`` and a
    ``before_tool_call`` callback can recover it into a guidance ``tool_result``
    instead of interrupting the turn. If no callback replaces the call, the
    placeholder raises a clear tool error rather than succeeding with an empty
    result.
    """
    tool_name, arguments = parse_tool_call(tool_call)
    tool_obj = tool_map.get(tool_name)
    if tool_obj is None:

        def raise_unknown_tool(**_: Any) -> None:
            raise BubError(ErrorKind.TOOL, f"Unknown tool name: {tool_name}.")

        return Tool(name=tool_name, handler=raise_unknown_tool), arguments
    return tool_obj, arguments


def parse_tool_call(tool_call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if tool_call.get("type") not in (None, "function"):
        raise BubError(ErrorKind.INVALID_INPUT, "Expected a function tool call with JSON object arguments.")
    function = tool_call.get("function") or {}
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise BubError(ErrorKind.INVALID_INPUT, "Expected a function tool call with a name.")
    arguments = function.get("arguments")
    if arguments is not None and not isinstance(arguments, str):
        raise BubError(ErrorKind.INVALID_INPUT, "Expected a function tool call with JSON object arguments.")
    try:
        parsed = TOOL_ARGUMENTS_ADAPTER.validate_json(arguments or "{}")
    except ValidationError as exc:
        raise BubError(ErrorKind.INVALID_INPUT, "Expected a function tool call with JSON object arguments.") from exc
    return name, parsed
