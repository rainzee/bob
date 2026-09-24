from __future__ import annotations

from typing import Any

import pytest
from loguru import logger
from pydantic import BaseModel

from bub.tools import Tool, model_tools, tool


class EchoInput(BaseModel):
    value: str


def test_tool_builds_completion_payload() -> None:
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    sample_tool = Tool(
        name="tests_sample_tool",
        description="Sample tool",
        parameters=parameters,
        handler=lambda value: value,
    )

    assert sample_tool.to_schema() == {
        "type": "function",
        "function": {
            "name": "tests_sample_tool",
            "description": "Sample tool",
            "parameters": parameters,
        },
    }


def test_model_tools_rewrites_dotted_names_without_mutating_original() -> None:
    tool_name = "tests.rename_me"

    @tool(name=tool_name, description="rename")
    def rename_me(value: str) -> str:
        return "ok"

    rewritten = model_tools([rename_me])

    assert [item.name for item in rewritten] == ["tests_rename_me"]
    assert rewritten[0].parameters == rename_me.parameters
    assert rename_me.name == tool_name
    assert "additionalProperties" not in rename_me.parameters


def test_model_tools_maps_dotted_names_to_model_aliases() -> None:
    dotted = Tool(name="tests.dotted", handler=lambda: None)
    plain = Tool(name="tests_plain", handler=lambda: None)

    rewritten = model_tools([dotted, plain])

    assert [item.name for item in rewritten] == ["tests_dotted", "tests_plain"]


@pytest.mark.asyncio
async def test_tool_decorator_preserves_metadata() -> None:
    tool_name = "tests.sync_tool"

    @tool(name=tool_name, description="Sync test tool", model=EchoInput)
    def sync_tool(payload: EchoInput) -> str:
        return payload.value.upper()

    assert sync_tool.name == tool_name
    assert sync_tool.description == "Sync test tool"
    assert await sync_tool.run(value="hello") == "HELLO"


@pytest.mark.asyncio
async def test_tool_wrapper_logs_and_omits_context_from_log_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_name = "tests.async_tool"
    messages: list[str] = []

    def record(message: str, *args: Any, **kwargs: Any) -> None:
        messages.append(message.format(*args, **kwargs))

    monkeypatch.setattr(logger, "info", record)

    @tool(name=tool_name, description="Async test tool", context=True)
    async def async_tool(value: str, context: object) -> str:
        return f"{value}:{context}"

    result = await async_tool.run("hello", context="ctx")

    assert result == "hello:ctx"
    assert len(messages) == 2
    assert messages[0] == 'tool.call.start name=tests.async_tool { "hello" }'
    assert messages[1].startswith("tool.call.success name=tests.async_tool elapsed_time=")


@pytest.mark.asyncio
async def test_tool_wrapper_logs_failures_before_reraising(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_name = "tests.failing_tool"
    errors: list[str] = []

    def record_exception(message: str, *args: Any, **kwargs: Any) -> None:
        errors.append(message.format(*args, **kwargs))

    monkeypatch.setattr(logger, "exception", record_exception)

    @tool(name=tool_name)
    def failing_tool() -> str:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await failing_tool.run()

    assert len(errors) == 1
    assert errors[0].startswith("tool.call.error name=tests.failing_tool elapsed_time=")


@pytest.mark.asyncio
async def test_tool_wrapper_reports_every_call_through_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_name = "tests.reported_tool"
    logged: list[str] = []

    def record_log(message: str, *args: Any, **kwargs: Any) -> None:
        logged.append(message.format(*args, **kwargs))

    monkeypatch.setattr(logger, "info", record_log)
    monkeypatch.setattr(logger, "exception", record_log)

    @tool(name=tool_name)
    def reported_tool(value: str) -> str:
        return value.upper()

    assert await reported_tool.run("hello") == "HELLO"

    assert logged[0] == f'tool.call.start name={tool_name} {{ "hello" }}'
    assert logged[1].startswith(f"tool.call.success name={tool_name} elapsed_time=")


@pytest.mark.asyncio
async def test_tool_direct_call_is_a_plain_value() -> None:
    tool_name = "tests.direct_call"

    def direct_call(value: str) -> str:
        return value.upper()

    direct_tool = tool(direct_call, name=tool_name)

    assert direct_tool.name == tool_name
    assert await direct_tool.run("hello") == "HELLO"
