from pathlib import Path
from unittest.mock import Mock

import pytest
from conftest import RecordingClient

from bub.agent import Agent
from bub.framework import BubFramework
from bub.hooks import Hooks
from bub.store import InMemoryTapeStore
from bub.streaming import AsyncStreamEvents, StreamEvent
from bub.tools import Tool, resolve_tool_names


def _reply() -> AsyncStreamEvents:
    async def events():
        yield StreamEvent("text", {"delta": "done"})
        yield StreamEvent("final", {"text": "done"})

    return AsyncStreamEvents(events())


async def recover_session_model(session_id: str, state: dict) -> dict | None:
    """A host-written load_state callback: restore the model this session last switched to

    Reading it from the agent's own tape is the whole mechanism; the library ships no
    opinion about which event names a session records.
    """

    agent = state.get("_runtime_agent")
    if not isinstance(agent, Agent):
        return None
    tape = agent.tape.session_tape(session_id, Path(state["_runtime_workspace"]))
    for entry in reversed(await tape.search(tape.query().kinds("event"))):
        if entry.payload.get("name") == "model_switch":
            model = (entry.payload.get("data") or {}).get("model")
            return {"model": model} if model else None
    return None


@pytest.fixture
def framework(tmp_path: Path) -> BubFramework:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)
    framework.add_hooks(Hooks(load_state=[recover_session_model]))
    return framework


@pytest.mark.asyncio
@pytest.mark.parametrize("has_saved_state", [False, True])
@pytest.mark.parametrize("override", [False, True])
async def test_a_store_is_isolated_and_an_explicit_override_wins(
    framework: BubFramework, has_saved_state: bool, override: bool
) -> None:
    other = Agent(framework, model="test:other", client=RecordingClient(), tools=[], tape_store=InMemoryTapeStore())
    other_tape = other.tape.session_tape("shared", framework.workspace)
    await other_tape.append_event("model_switch", {"model": "test:other"})

    agent = Agent(framework, model="test:model", client=RecordingClient(), tools=[], tape_store=InMemoryTapeStore())
    tape = agent.tape.session_tape("shared", framework.workspace)
    if has_saved_state:
        await tape.append_event("model_switch", {"model": "test:saved"})

    runner = Mock(side_effect=lambda **kwargs: _reply())
    agent.model_runner.run = runner
    stream = await agent.run_stream(
        session_id="shared",
        prompt="hello",
        model="test:explicit" if override else None,
    )

    assert [event.kind async for event in stream] == ["text", "final"]
    call = runner.call_args.kwargs
    expected = "test:saved" if has_saved_state else agent.model
    assert call["model"] == ("test:explicit" if override else expected)
    assert call["tape"].context.state["_runtime_agent"] is agent


def test_instance_tool_names_resolve_aliases_and_exclusions_from_one_index() -> None:
    names = ["sdk.lookup", "sdk.other"]
    assert resolve_tool_names([" SDK_LOOKUP "], all_names=iter(names)) == {"sdk.lookup"}
    assert resolve_tool_names(None, exclude=["SDK_OTHER"], all_names=iter(names)) == {"sdk.lookup"}
    assert resolve_tool_names(["sdk_lookup"], exclude=["sdk.lookup"], all_names=names) == set()
    assert resolve_tool_names(None, all_names=[]) == set()
    with pytest.raises(ValueError, match="bash"):
        resolve_tool_names(["bash"], all_names=names)
    with pytest.raises(ValueError, match="bash"):
        resolve_tool_names(None, exclude=["bash"], all_names=names)


@pytest.mark.asyncio
async def test_agent_allowlist_accepts_unregistered_instance_tool(framework: BubFramework) -> None:
    tool = Tool.from_callable(lambda: "found", name="sdk.lookup")
    agent = Agent(
        framework,
        model="test:model",
        client=RecordingClient(),
        tools=[tool],
        tape_store=InMemoryTapeStore(),
    )
    runner = Mock(side_effect=lambda **kwargs: _reply())
    agent.model_runner.run = runner

    stream = await agent.run_stream(session_id="sdk", prompt="lookup", allowed_tools=[" SDK_LOOKUP "])

    assert [event.kind async for event in stream] == ["text", "final"]
    assert [tool.name for tool in runner.call_args.kwargs["tools"]] == ["sdk_lookup"]
