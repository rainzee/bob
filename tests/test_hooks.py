"""Hooks 的槽位语义: 顺序, 拼接, 同步异步, 失败即抛"""

from __future__ import annotations

from typing import Any

import pytest

from bub.hooks import Hooks, LlmCallDecision, LlmCallRequest, ToolCall, ToolCallDecision


def request() -> LlmCallRequest:
    return LlmCallRequest(run_id="run-1", model="openai:gpt-x", messages=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_load_state_runs_in_sequence_and_later_keys_win() -> None:
    seen: list[dict[str, Any]] = []

    def low(session_id: str, state: dict[str, Any]) -> dict[str, Any]:
        seen.append(dict(state))
        return {"from": "low", "kept": session_id}

    async def high(session_id: str, state: dict[str, Any]) -> dict[str, Any]:
        seen.append(dict(state))
        return {"from": "high"}

    hooks = Hooks(load_state=[low, high])
    state = await hooks.run_load_state("session-1", {"seed": 1})

    assert state == {"seed": 1, "from": "high", "kept": "session-1"}
    assert seen == [{"seed": 1}, {"seed": 1, "from": "low", "kept": "session-1"}]


@pytest.mark.asyncio
async def test_load_state_ignores_empty_updates() -> None:
    hooks = Hooks(load_state=[lambda session_id, state: None, lambda session_id, state: {}])

    assert await hooks.run_load_state("s", {"seed": 1}) == {"seed": 1}


@pytest.mark.asyncio
async def test_system_prompt_joins_blocks_in_sequence_and_skips_empties() -> None:
    def low(prompt, state) -> str:
        return "low"

    async def high(prompt, state) -> str | None:
        return "high"

    def empty(prompt, state) -> str | None:
        return None

    hooks = Hooks(system_prompt=[low, high, empty])

    assert await hooks.run_system_prompt("hello", {}) == "low\n\nhigh"


@pytest.mark.asyncio
async def test_before_llm_call_chains_in_sequence() -> None:
    async def swap(request: LlmCallRequest, state: dict) -> LlmCallRequest:
        from dataclasses import replace

        return replace(request, model="anthropic:claude")

    def append(request: LlmCallRequest, state: dict) -> LlmCallRequest:
        from dataclasses import replace

        assert request.model == "anthropic:claude"
        return replace(request, messages=[*request.messages, {"role": "user", "content": "extra"}])

    result, decision = await Hooks(before_llm_call=[swap, append]).run_before_llm_call(request(), {})

    assert decision is None
    assert result.model == "anthropic:claude"
    assert result.messages[-1]["content"] == "extra"


@pytest.mark.asyncio
async def test_before_llm_call_finish_short_circuits() -> None:
    def limit(request: LlmCallRequest, state: dict) -> LlmCallDecision:
        return LlmCallDecision.finish("call budget exhausted")

    def later(request: LlmCallRequest, state: dict) -> None:
        raise AssertionError("must not run after finish")

    _request, decision = await Hooks(before_llm_call=[limit, later]).run_before_llm_call(request(), {})

    assert decision is not None
    assert decision.text == "call budget exhausted"


@pytest.mark.asyncio
async def test_before_tool_call_proceed_folds_arguments_forward() -> None:
    def rewrite(call: ToolCall, state: dict) -> ToolCallDecision:
        return ToolCallDecision.proceed(arguments={**call.arguments, "safe": True})

    def verify(call: ToolCall, state: dict) -> None:
        assert call.arguments["safe"] is True
        return None

    call, decision = await Hooks(before_tool_call=[rewrite, verify]).run_before_tool_call(
        ToolCall(run_id="run-1", tool="shell", arguments={"cmd": "ls"}), {}
    )

    assert decision.action == "proceed"
    assert call.arguments == {"cmd": "ls", "safe": True}


@pytest.mark.asyncio
async def test_before_tool_call_deny_short_circuits() -> None:
    seen: list[str] = []

    def deny(call: ToolCall, state: dict) -> ToolCallDecision:
        return ToolCallDecision.deny("dangerous command")

    def later(call: ToolCall, state: dict) -> None:
        seen.append(call.tool)
        return None

    _call, decision = await Hooks(before_tool_call=[deny, later]).run_before_tool_call(
        ToolCall(run_id="run-1", tool="shell", arguments={}), {}
    )

    assert decision.action == "deny"
    assert decision.message == "dangerous command"
    assert seen == []


@pytest.mark.asyncio
async def test_callbacks_may_be_sync_or_async() -> None:
    seen: list[str] = []

    def sync_observer(call: ToolCall, result, state: dict) -> None:
        seen.append("sync")

    async def async_observer(call: ToolCall, result, state: dict) -> None:
        seen.append("async")

    from bub.hooks import ToolCallResult

    outcome = ToolCallResult(run_id="run-1", tool="shell", arguments={}, result="ok")
    await Hooks(after_tool_call=[sync_observer, async_observer]).run_after_tool_call(
        ToolCall(run_id="run-1", tool="shell", arguments={}), outcome, {}
    )

    assert seen == ["sync", "async"]


def test_add_concatenates_each_slot_in_order() -> None:
    def first(session_id, state):
        return None

    def second(session_id, state):
        return None

    combined = Hooks(load_state=[first]) + Hooks(load_state=[second])

    assert combined.load_state == (first, second)
    assert combined.system_prompt == ()


@pytest.mark.asyncio
async def test_callback_failures_propagate() -> None:
    def boom(session_id: str, state: dict) -> None:
        raise RuntimeError("host callback broke")

    with pytest.raises(RuntimeError, match="host callback broke"):
        await Hooks(load_state=[boom]).run_load_state("s", {})


@pytest.mark.asyncio
async def test_bad_return_types_are_rejected() -> None:
    def bad_llm(request: LlmCallRequest, state: dict) -> str:
        return "not-a-request"

    def bad_tool(call: ToolCall, state: dict) -> str:
        return "not-a-decision"

    with pytest.raises(TypeError, match="before_llm_call must return"):
        await Hooks(before_llm_call=[bad_llm]).run_before_llm_call(request(), {})
    with pytest.raises(TypeError, match="before_tool_call must return"):
        await Hooks(before_tool_call=[bad_tool]).run_before_tool_call(
            ToolCall(run_id="run-1", tool="shell", arguments={}), {}
        )
