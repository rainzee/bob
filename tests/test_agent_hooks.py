"""Agent-loop interception semantics: chaining, short-circuit, fail-fast (issue #253)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest

from bub.errors import BubError
from bub.hooks import (
    Hooks,
    LlmCallRequest,
    LlmCallResult,
    ToolCall,
    ToolCallDecision,
    ToolCallResult,
)
from bub.tools import Tool, ToolExecutor


def make_hooks(
    *,
    before_llm_call: Sequence[Any] = (),
    after_llm_call: Sequence[Any] = (),
    before_tool_call: Sequence[Any] = (),
    after_tool_call: Sequence[Any] = (),
) -> Hooks:
    return Hooks(
        before_llm_call=before_llm_call,
        after_llm_call=after_llm_call,
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
    )


def request() -> LlmCallRequest:
    return LlmCallRequest(run_id="run-1", model="openai:gpt-x", messages=[{"role": "user", "content": "hi"}])


class TestBeforeLlmCall:
    @pytest.mark.asyncio
    async def test_chain_folds_modifications_in_order(self) -> None:
        def swap(request: LlmCallRequest, state: dict) -> LlmCallRequest:
            return replace(request, model="anthropic:claude")

        def append(request: LlmCallRequest, state: dict) -> LlmCallRequest:
            # must see swap's change (sequence-order chaining)
            assert request.model == "anthropic:claude"
            return replace(request, messages=[*request.messages, {"role": "user", "content": "extra"}])

        hooks = make_hooks(before_llm_call=[swap, append])
        result, decision = await hooks.run_before_llm_call(request(), {})

        assert decision is None
        assert result.model == "anthropic:claude"
        assert result.messages[-1]["content"] == "extra"

    @pytest.mark.asyncio
    async def test_none_returns_leave_request_unchanged(self) -> None:
        def noop(request: LlmCallRequest, state: dict) -> None:
            return None

        hooks = make_hooks(before_llm_call=[noop])
        original = request()

        assert await hooks.run_before_llm_call(original, {}) == (original, None)

    @pytest.mark.asyncio
    async def test_raising_callback_propagates(self) -> None:
        def boom(request: LlmCallRequest, state: dict) -> LlmCallRequest:
            raise RuntimeError("host callback broke")

        def after(request: LlmCallRequest, state: dict) -> LlmCallRequest:
            return replace(request, model="fallback:model")

        hooks = make_hooks(before_llm_call=[boom, after])

        with pytest.raises(RuntimeError, match="host callback broke"):
            await hooks.run_before_llm_call(request(), {})


class TestBeforeToolCall:
    @pytest.mark.asyncio
    async def test_deny_short_circuits_remaining_callbacks(self) -> None:
        seen: list[str] = []

        def deny(call: ToolCall, state: dict) -> ToolCallDecision:
            return ToolCallDecision.deny("dangerous command")

        def later(call: ToolCall, state: dict) -> None:
            seen.append(call.tool)
            return None

        hooks = make_hooks(before_tool_call=[deny, later])
        _, decision = await hooks.run_before_tool_call(ToolCall(run_id="run-1", tool="shell", arguments={}), {})

        assert decision.action == "deny"
        assert decision.message == "dangerous command"
        assert seen == []


class TestToolExecutorIntegration:
    def tool(self) -> Tool:
        def handler(cmd: str) -> str:
            return f"ran:{cmd}"

        return Tool(name="shell", handler=handler, description="", parameters={})

    @pytest.mark.asyncio
    async def test_deny_surfaces_tool_error_result(self) -> None:
        def deny(call: ToolCall, state: dict) -> ToolCallDecision:
            return ToolCallDecision.deny("blocked by policy")

        executor = ToolExecutor(hooks=make_hooks(before_tool_call=[deny]))
        execution = await executor.execute_async([(self.tool(), {"cmd": "rm -rf /"})])

        assert execution.error is not None
        assert "blocked by policy" in execution.error.message
        assert execution.tool_results[0]["message"] == "blocked by policy"

    @pytest.mark.asyncio
    async def test_replace_skips_handler(self) -> None:
        def replace_result(call: ToolCall, state: dict) -> ToolCallDecision:
            return ToolCallDecision.replace("cached result")

        executor = ToolExecutor(hooks=make_hooks(before_tool_call=[replace_result]))
        execution = await executor.execute_async([(self.tool(), {"cmd": "ls"})])

        assert execution.error is None
        assert execution.tool_results == ["cached result"]

    @pytest.mark.asyncio
    async def test_after_tool_call_observes_success_and_error(self) -> None:
        observed: list[ToolCallResult] = []

        def observe(call: ToolCall, result: ToolCallResult, state: dict) -> None:
            observed.append(result)

        def failing(cmd: str) -> str:
            raise ValueError("nope")

        executor = ToolExecutor(hooks=make_hooks(after_tool_call=[observe]))
        await executor.execute_async([(self.tool(), {"cmd": "ls"})])
        await executor.execute_async([(Tool(name="bad", handler=failing, description="", parameters={}), {"cmd": "x"})])

        assert observed[0].result == "ran:ls"
        assert observed[0].error is None
        assert isinstance(observed[1].error, BubError)  # original error object, kind/details preserved
        assert observed[1].error.kind is not None
        assert "bad" in observed[1].tool

    @pytest.mark.asyncio
    async def test_after_tool_call_can_replace_the_result_seen_by_the_model(self) -> None:
        def bound(call: ToolCall, result: ToolCallResult, state: dict) -> None:
            if isinstance(result.result, str):
                result.result = f"bounded:{result.result}"

        executor = ToolExecutor(hooks=make_hooks(after_tool_call=[bound]))
        execution = await executor.execute_async([(self.tool(), {"cmd": "ls"})])

        assert execution.tool_results == ["bounded:ran:ls"]

    @pytest.mark.asyncio
    async def test_modified_arguments_reach_handler(self) -> None:
        def rewrite(call: ToolCall, state: dict) -> ToolCallDecision:
            return ToolCallDecision.proceed(arguments={"cmd": "safe-ls"})

        executor = ToolExecutor(hooks=make_hooks(before_tool_call=[rewrite]))
        execution = await executor.execute_async([(self.tool(), {"cmd": "rm"})])

        assert execution.tool_results == ["ran:safe-ls"]

    @pytest.mark.asyncio
    async def test_no_hooks_keeps_current_behavior(self) -> None:
        execution = await ToolExecutor().execute_async([(self.tool(), {"cmd": "ls"})])

        assert execution.tool_results == ["ran:ls"]


class TestAfterLlmCall:
    @pytest.mark.asyncio
    async def test_observe_only(self) -> None:
        observed: list[LlmCallResult] = []

        def observe(request: LlmCallRequest, result: LlmCallResult, state: dict) -> None:
            observed.append(result)

        hooks = make_hooks(after_llm_call=[observe])
        result = LlmCallResult(run_id="run-1", text="hello", usage={"total_tokens": 5}, duration_ms=12)

        await hooks.run_after_llm_call(request(), result, {})

        assert observed == [result]


class TestModelRunnerHookIntegration:
    """Regression tests for PR #255 review findings (effective request, exactly-once)."""

    def _runner_and_tape(self, hooks: Hooks, captured: dict):
        from bub.builtin.model_runner import ModelRunner
        from bub.store import AsyncTapeStoreAdapter, InMemoryTapeStore
        from bub.tape import Tape, TapeContext

        class FakeRunner(ModelRunner):
            async def completion_response(self, *, model, messages, tools, max_tokens=None, reasoning_effort=None):
                captured.update(model=model, max_tokens=max_tokens)

                async def chunks():
                    return
                    yield  # pragma: no cover

                return chunks()

        runner = FakeRunner(model="openai:orig", max_tokens=100, hooks=hooks)
        store = AsyncTapeStoreAdapter(InMemoryTapeStore())
        tape = Tape(store, TapeContext(anchor=None)).scoped("t1")
        return runner, tape

    @pytest.mark.asyncio
    async def test_rewritten_model_and_max_tokens_reach_provider_and_tape(self) -> None:
        def reroute(request: LlmCallRequest, state: dict) -> LlmCallRequest:
            return replace(request, model="anthropic:new", max_tokens=42)

        captured: dict = {}
        runner, tape = self._runner_and_tape(make_hooks(before_llm_call=[reroute]), captured)
        events = runner.run(tape=tape, model="openai:orig", tools=[], system_prompt=None, prompt="hi")
        async for _ in events:
            pass

        assert captured == {"model": "anthropic:new", "max_tokens": 42}
        entries = list(await tape.store.fetch_all(tape.query().kinds("event")))
        run_events = [e for e in entries if e.payload.get("name") == "run"]
        assert run_events[-1].payload["data"]["model"] == "anthropic:new"

    @pytest.mark.asyncio
    async def test_after_llm_call_not_fired_on_early_close(self) -> None:
        observed: list[LlmCallResult] = []

        def observe(request: LlmCallRequest, result: LlmCallResult, state: dict) -> None:
            observed.append(result)

        captured: dict = {}
        runner, tape = self._runner_and_tape(make_hooks(after_llm_call=[observe]), captured)

        async def fake_events(completion, state, output):
            from bub.streaming import StreamEvent

            yield StreamEvent("text", {"delta": "a"})
            yield StreamEvent("text", {"delta": "b"})

        runner._completion_events = fake_events  # type: ignore[method-assign]
        events = runner.run(tape=tape, model="openai:orig", tools=[], system_prompt=None, prompt="hi")
        iterator = events.__aiter__()
        await iterator.__anext__()
        await iterator.aclose()

        # Consumer close is intentionally NOT a terminal observation:
        # after_llm_call fires only for real completions and Exception failures.
        assert observed == []

    @pytest.mark.asyncio
    async def test_after_llm_call_fires_exactly_once_on_success(self) -> None:
        observed: list[LlmCallResult] = []

        def observe(request: LlmCallRequest, result: LlmCallResult, state: dict) -> None:
            observed.append(result)

        captured: dict = {}
        runner, tape = self._runner_and_tape(make_hooks(after_llm_call=[observe]), captured)
        events = runner.run(tape=tape, model="openai:orig", tools=[], system_prompt=None, prompt="hi")
        async for _ in events:
            pass

        assert len(observed) == 1
        assert observed[0].error is None


class TestToolCancellation:
    @pytest.mark.asyncio
    async def test_after_tool_call_not_fired_on_cancel(self) -> None:
        import asyncio

        observed: list[ToolCallResult] = []

        def observe(call: ToolCall, result: ToolCallResult, state: dict) -> None:
            observed.append(result)

        started = asyncio.Event()

        async def blocking(cmd: str) -> str:
            started.set()
            await asyncio.Event().wait()  # blocks until cancelled
            return "unreachable"

        executor = ToolExecutor(hooks=make_hooks(after_tool_call=[observe]))
        task = asyncio.create_task(
            executor.execute_async([
                (Tool(name="block", handler=blocking, description="", parameters={}), {"cmd": "x"})
            ])
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Cancellation is intentionally NOT a terminal observation:
        # after_tool_call fires only for success, failure and deny/replace.
        assert observed == []
