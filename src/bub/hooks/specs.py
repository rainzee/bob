"""Pluggy namespace and extension specifications for Bub hooks."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pluggy

from bub.hooks.interception import (
    LlmCallDecision,
    LlmCallRequest,
    LlmCallResult,
    ToolCall,
    ToolCallDecision,
    ToolCallResult,
)
from bub.sidecars import TapeSidecar
from bub.store import AsyncTapeStore, TapeStore
from bub.tape import TapeContext
from bub.turn import TurnState

BUB_HOOK_NAMESPACE = "bub"
hookspec = pluggy.HookspecMarker(BUB_HOOK_NAMESPACE)
hookimpl = pluggy.HookimplMarker(BUB_HOOK_NAMESPACE)


class BubHookSpecs:
    """Hook contract for Bub framework extensions."""

    @hookspec
    def load_state(self, session_id: str, state: TurnState) -> TurnState:
        """Load a partial state snapshot for one session.

        The framework passes the state it has accumulated so far, so an
        implementation may read values a higher-priority hook contributed.
        """
        raise NotImplementedError

    @hookspec
    def before_llm_call(self, request: LlmCallRequest, state: TurnState) -> LlmCallRequest | LlmCallDecision | None:
        """Observe, modify or short-circuit an outgoing agent-loop LLM request.

        Implementations are chained in pluggy's LIFO order (last registered
        runs first): each receives the
        request as modified by earlier implementations, and may return a
        modified copy (``dataclasses.replace``), ``None`` to leave it
        unchanged, or ``LlmCallDecision.finish(text)`` to skip the provider
        call and emit ``text`` as the final output (cost guards / call
        limits). Exceptions are logged and skipped, never fatal.
        """

    @hookspec
    def after_llm_call(self, request: LlmCallRequest, result: LlmCallResult, state: TurnState) -> None:
        """Observe the terminal outcome of one agent-loop LLM call.

        Fires exactly once per completed call — success (for streaming
        completions, after the stream is fully consumed) or ``Exception``
        failure (``result.error`` set). Cancellation and consumer close
        (``BaseException``) intentionally bypass this hook. Return values
        are ignored; exceptions are logged and skipped.
        """

    @hookspec
    def before_tool_call(self, call: ToolCall, state: TurnState) -> ToolCallDecision | None:
        """Decide whether/how one tool invocation runs.

        Return ``None`` or ``ToolCallDecision.proceed(...)`` to continue
        (optionally with modified arguments, visible to later
        implementations), ``ToolCallDecision.replace(result)`` to skip the
        tool and use ``result``, or ``ToolCallDecision.deny(message)`` to
        surface a tool error instead. ``replace``/``deny`` short-circuit
        remaining implementations. Blocking is only possible via the
        decision object — exceptions are logged and skipped.
        """

    @hookspec
    def after_tool_call(self, call: ToolCall, result: ToolCallResult, state: TurnState) -> None:
        """Handle the terminal outcome of one tool invocation.

        Fires for success, failure (``result.error`` set), denial and
        replacement. An implementation may replace the model-facing value by
        assigning ``result.result``; for failures, ``result.error`` remains
        set. Return values are ignored; exceptions are logged.
        """

    @hookspec
    def system_prompt(self, prompt: str | list[dict], state: TurnState) -> str:
        """Provide a system prompt to be prepended to all model prompts."""
        raise NotImplementedError

    @hookspec(firstresult=True)
    def provide_tape_store(self) -> TapeStore | AsyncTapeStore | None:
        """Provide a tape store instance for Bub's conversation recording feature."""
        raise NotImplementedError

    @hookspec
    def provide_lifespan(self) -> AsyncIterator[None] | Iterator[None] | None:
        """Yield once to own resources for the duration of framework.running()."""

    @hookspec
    def provide_tape_sidecar(self) -> TapeSidecar:
        """Provide a capability backed by a sibling tape mounted on every session tape."""
        raise NotImplementedError

    @hookspec(firstresult=True)
    def build_tape_context(self) -> TapeContext:
        """Build a tape context for the current session, to be used to build context messages."""
        raise NotImplementedError
