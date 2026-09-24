from __future__ import annotations

import json
from pathlib import Path

import pytest

from bub.builtin.context import default_tape_context
from bub.builtin.spill import (
    SPILL_READ_MODEL_NAME,
    SPILL_READ_TOOL_NAME,
    SPILL_SIDECAR_NAME,
    SpillStore,
    spill_read,
    spill_tool_result,
)
from bub.hooks import Hooks, ToolCall, ToolCallResult
from bub.store import AsyncTapeStoreAdapter, FileTapeStore, InMemoryTapeStore, TapeStore
from bub.tape import Tape
from bub.tools import Tool, ToolContext, ToolExecutor, model_tools, render_tools_prompt


def _spill_executor() -> ToolExecutor:
    return ToolExecutor(hooks=Hooks(after_tool_call=[spill_tool_result]))


def _handle_from_ref(ref: str) -> str:
    return ref.split("handle: ", 1)[1].split("]", 1)[0]


def _page_content(page: str) -> str:
    return page.split("content:\n", 1)[1]


def _page_field(page: str, name: str) -> str:
    prefix = f"{name}: "
    return next(line.removeprefix(prefix) for line in page.splitlines() if line.startswith(prefix))


def _root_tape(store: TapeStore, *, threshold: int = 1) -> Tape:
    spill = SpillStore(threshold=threshold)
    return Tape(AsyncTapeStoreAdapter(store), default_tape_context(), sidecars=(spill,)).scoped("session")


async def _read_page(
    context: ToolContext,
    handle: str,
    *,
    cursor: int = 0,
    count: int = 1,
    from_end: bool = False,
) -> str:
    execution = await _spill_executor().execute_async(
        [(spill_read, {"handle": handle, "cursor": cursor, "count": count, "from_end": from_end})],
        context=context,
    )
    page = execution.tool_results[0]
    assert isinstance(page, str)
    return page


@pytest.mark.asyncio
async def test_oversized_result_is_bounded_and_readable_across_merge(tmp_path: Path) -> None:
    parent = FileTapeStore(tmp_path / "tapes")
    root = _root_tape(parent)
    output = ("alpha🙂beta\n" * 5000) + "the-end"

    async with root.fork_tape() as tape:
        context = ToolContext(tape=tape, run_id="run-1")
        tool = Tool(name="large", handler=lambda: output)
        execution = await _spill_executor().execute_async([(tool, {})], context=context)

        ref = execution.tool_results[0]
        assert isinstance(ref, str)
        assert "tool output spilled" in ref
        assert len(ref) < 2000
        handle = _handle_from_ref(ref)

        cursor = 0
        restored: list[str] = []
        while True:
            page = await _read_page(context, handle, cursor=cursor, count=2)
            restored.append(_page_content(page))
            if _page_field(page, "complete") == "true":
                break
            cursor = int(_page_field(page, "next_cursor"))

        assert "".join(restored) == output

        tail = await _read_page(context, handle, from_end=True)
        assert _page_content(tail).endswith("the-end")

        await tape.record_chat(
            run_id="run-1",
            system_prompt=None,
            new_messages=[],
            response_text=None,
            tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "large", "arguments": "{}"}}],
            tool_results=execution.tool_results,
        )
        request_messages = await tape.read_messages()
        request_body = json.dumps(request_messages, ensure_ascii=False)
        assert handle in request_body
        assert output not in request_body

    persisted_context = ToolContext(tape=root, run_id="run-2")
    persisted = await _read_page(persisted_context, handle)
    assert _page_content(persisted) == restored[0][: len(_page_content(persisted))]


@pytest.mark.asyncio
async def test_spill_configuration_preserves_results_that_should_not_be_spilled(tmp_path: Path) -> None:
    parent = InMemoryTapeStore()
    root = _root_tape(parent, threshold=100)

    async with root.fork_tape() as tape:
        context = ToolContext(tape=tape, run_id="run-1")
        small_results = ["tiny", {"value": "tiny"}, ["tiny"]]
        small = await _spill_executor().execute_async(
            [
                (Tool(name=f"small-{index}", handler=lambda result=result: result), {})
                for index, result in enumerate(small_results)
            ],
            context=context,
        )

        assert small.tool_results == small_results

        def fail() -> None:
            raise RuntimeError("small failure")

        small_error = await _spill_executor().execute_async([(Tool(name="failing", handler=fail), {})], context=context)
        assert small_error.error is not None
        assert small_error.tool_results == [small_error.error.as_dict()]

    disabled = _root_tape(parent, threshold=0).scoped("disabled")
    async with disabled.fork_tape() as tape:
        execution = await _spill_executor().execute_async(
            [(Tool(name="large", handler=lambda: "x" * 20_000), {})],
            context=ToolContext(tape=tape, run_id="run-2"),
        )
    assert execution.tool_results == ["x" * 20_000]


@pytest.mark.parametrize("output", [{"value": "x" * 20_000}, ["x" * 20_000]])
@pytest.mark.asyncio
async def test_oversized_structured_result_is_spilled_as_json(tmp_path: Path, output: object) -> None:
    root = _root_tape(InMemoryTapeStore(), threshold=100)

    async with root.fork_tape() as tape:
        context = ToolContext(tape=tape, run_id="run-1")
        execution = await _spill_executor().execute_async(
            [(Tool(name="structured", handler=lambda: output), {})], context=context
        )

        ref = execution.tool_results[0]
        assert isinstance(ref, str)
        assert "tool output spilled" in ref

        handle = _handle_from_ref(ref)
        page = await _read_page(context, handle, count=4)
        assert json.loads(_page_content(page)) == output


@pytest.mark.asyncio
async def test_spill_runs_after_other_result_hooks(tmp_path: Path) -> None:
    def expand_result(call: ToolCall, result: ToolCallResult, state: dict) -> None:
        result.result = {"value": "x" * 20_000}

    executor = ToolExecutor(hooks=Hooks(after_tool_call=[expand_result, spill_tool_result]))
    root = _root_tape(InMemoryTapeStore(), threshold=100)

    async with root.fork_tape() as tape:
        execution = await executor.execute_async(
            [(Tool(name="expanded", handler=lambda: "small"), {})],
            context=ToolContext(tape=tape, run_id="run-1"),
        )

    assert "tool output spilled" in execution.tool_results[0]


@pytest.mark.parametrize(
    ("error_message", "replacement", "should_spill"),
    [
        ("original secret " * 1000, "sanitized failure", False),
        ("small failure", "replacement " * 2000, True),
    ],
)
@pytest.mark.asyncio
async def test_failure_replacement_is_used_for_spill_check(
    tmp_path: Path,
    error_message: str,
    replacement: str,
    should_spill: bool,
) -> None:
    def replace_failure(call: ToolCall, result: ToolCallResult, state: dict) -> None:
        if result.error is not None:
            result.result = replacement

    executor = ToolExecutor(hooks=Hooks(after_tool_call=[replace_failure, spill_tool_result]))
    root = _root_tape(InMemoryTapeStore(), threshold=100)

    def fail() -> None:
        raise RuntimeError(error_message)

    async with root.fork_tape() as tape:
        context = ToolContext(tape=tape, run_id="run-1")
        execution = await executor.execute_async([(Tool(name="bash", handler=fail), {})], context=context)

        assert execution.error is not None
        result = execution.tool_results[0]
        if should_spill:
            assert isinstance(result, str)
            assert "tool output spilled" in result
            page = await _read_page(context, _handle_from_ref(result), count=4)
            assert _page_content(page) == replacement
        else:
            assert result == replacement


@pytest.mark.asyncio
async def test_oversized_tool_error_is_spilled_and_remains_a_failure(tmp_path: Path) -> None:
    root = _root_tape(InMemoryTapeStore(), threshold=100)
    error_message = "failure detail " * 1000

    def fail() -> None:
        raise RuntimeError(error_message)

    async with root.fork_tape() as tape:
        context = ToolContext(tape=tape, run_id="run-1")
        execution = await _spill_executor().execute_async([(Tool(name="failing", handler=fail), {})], context=context)

        assert execution.error is not None
        ref = execution.tool_results[0]
        assert isinstance(ref, str)
        assert "tool output spilled" in ref
        assert error_message not in ref

        handle = _handle_from_ref(ref)
        page = await _read_page(context, handle, count=4)
        assert json.loads(_page_content(page)) == execution.error.as_dict()


@pytest.mark.asyncio
async def test_temporary_fork_discards_spilled_content(tmp_path: Path) -> None:
    parent = InMemoryTapeStore()
    root = _root_tape(parent)

    async with root.fork_tape(merge_back=False) as tape:
        context = ToolContext(tape=tape, run_id="run-1")
        execution = await _spill_executor().execute_async(
            [(Tool(name="large", handler=lambda: "x" * 20_000), {})], context=context
        )
        handle = _handle_from_ref(execution.tool_results[0])
        assert "content:" in await _read_page(context, handle)

    missing = await _read_page(ToolContext(tape=root), handle)
    assert "no spilled tool result" in missing


@pytest.mark.asyncio
async def test_unknown_handle_and_invalid_read_bounds_are_friendly(tmp_path: Path) -> None:
    root = _root_tape(InMemoryTapeStore())
    context = ToolContext(tape=root)

    assert "no spilled tool result" in await spill_read.run(handle="missing", context=context)
    assert await spill_read.run(handle="missing", cursor=-1, context=context) == "`cursor` must be >= 0."
    assert await spill_read.run(handle="missing", count=0, context=context) == "`count` must be >= 1."


def test_spill_read_uses_the_builtin_tool_naming_convention() -> None:
    assert spill_read.name == SPILL_READ_TOOL_NAME == "spill.read"
    assert model_tools([spill_read])[0].name == SPILL_READ_MODEL_NAME == "spill_read"
    assert "spill_read(handle, cursor?, count?, from_end?)" in render_tools_prompt([spill_read])


@pytest.mark.asyncio
async def test_tape_reset_clears_spilled_results_from_every_mounted_tape() -> None:
    parent = InMemoryTapeStore()
    root = _root_tape(parent)
    sidecar = root.sidecar_tape_name(SPILL_SIDECAR_NAME)
    await root.ensure_bootstrap_anchor()

    async with root.fork_tape() as tape:
        execution = await _spill_executor().execute_async(
            [(Tool(name="large", handler=lambda: "archived output\n" * 5000), {})],
            context=ToolContext(tape=tape, run_id="run-1"),
        )
        ref = execution.tool_results[0]
        assert isinstance(ref, str)
        handle = _handle_from_ref(ref)
        await tape.record_chat(
            run_id="run-1",
            system_prompt=None,
            new_messages=[{"role": "user", "content": "clear this"}],
            response_text=None,
            tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "large", "arguments": "{}"}}],
            tool_results=execution.tool_results,
        )

    assert parent.read(sidecar) is not None
    await root.reset()

    main_entries = parent.read("session") or []
    assert [entry.kind for entry in main_entries] == ["anchor", "event", "event"]
    assert [entry.payload.get("name") for entry in main_entries if entry.kind == "event"] == [
        "handoff",
        "sidecar.reset",
    ]
    assert parent.read(sidecar) is None
    assert "no spilled tool result" in await _read_page(ToolContext(tape=root), handle)
