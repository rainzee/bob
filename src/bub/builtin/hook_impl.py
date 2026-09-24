from collections.abc import AsyncIterator
from datetime import datetime
from difflib import get_close_matches
from typing import Any, cast

from bub.builtin.agent import Agent
from bub.builtin.context import default_tape_context, render_tool_result
from bub.envelope import content_of, field_of
from bub.errors import BubError
from bub.framework import BubFramework
from bub.hooks import hookimpl
from bub.hooks.interception import ToolCall, ToolCallDecision, ToolCallResult
from bub.message import MediaItem, Message, audio_format_from_mime_type
from bub.sidecars import TapeSidecar
from bub.store import TapeStore
from bub.streaming import AsyncStreamEvents, StreamState
from bub.tape import Tape, TapeContext
from bub.turn import TurnState
from bub.utils import workspace_from_state

AGENTS_FILE_NAME = "AGENTS.md"
DEFAULT_SYSTEM_PROMPT = """\
<general_instruct>
Call tools or skills to finish the task.
</general_instruct>
<context_contract>
Excessively long context may cause model call failures. In this case, you MAY use tape.info to retrieve the token usage and you SHOULD use tape.handoff tool to shorten the retrieved history.
</context_contract>
"""
DEFAULT_CONTINUE_PROMPT = "Continue the task until all targets are completed."


def _input_audio_part(data_url: str, mime_type: str) -> dict[str, Any] | None:
    prefix, separator, data = data_url.partition("base64,")
    if not separator or not prefix.startswith("data:audio/") or not data:
        return None
    return {
        "type": "input_audio",
        "input_audio": {"data": data, "format": audio_format_from_mime_type(mime_type)},
    }


class BuiltinImpl:
    """Default hook implementations for the turn pipeline"""

    def __init__(self, framework: BubFramework) -> None:
        self.framework = framework
        self._agent: Agent | None = None

    def _get_agent(self, state: TurnState | None = None) -> Agent:
        if state and "_runtime_agent" in state:
            return cast("Agent", state["_runtime_agent"])
        if self._agent is None:
            self._agent = Agent(self.framework)
        return self._agent

    async def _recover_session_model(self, session_id: str, *, agent: Agent) -> str | None:
        """Recover the latest per-session model override recorded on the session tape.

        The ``model`` tool records each switch as a ``model_switch`` event on the
        session's tape. Scanning that tape here (before the per-turn fork exists)
        reads the persisted store, so a choice from a prior turn or restart is
        restored. Returns ``None`` when nothing was recorded, so a fresh session
        never inherits another session's model.
        """
        session = agent.tape.session_tape(session_id, self.framework.workspace)
        entries = list(await session.store.fetch_all(session.query().kinds("event")))
        for entry in reversed(entries):
            if entry.kind == "event" and entry.payload.get("name") == "model_switch":
                model = (entry.payload.get("data") or {}).get("model")
                return str(model) if model else None
        return None

    async def _recover_session_reasoning_effort(self, session_id: str, *, agent: Agent) -> str | None:
        """Recover the latest per-session reasoning effort override."""
        session = agent.tape.session_tape(session_id, self.framework.workspace)
        entries = list(await session.store.fetch_all(session.query().kinds("event")))
        for entry in reversed(entries):
            if entry.kind == "event" and entry.payload.get("name") == "reasoning_effort_switch":
                reasoning_effort = (entry.payload.get("data") or {}).get("reasoning_effort")
                return str(reasoning_effort) if reasoning_effort else None
        return None

    @hookimpl
    def resolve_session(self, message: Message) -> str:
        session_id = field_of(message, "session_id")
        if session_id is not None and str(session_id).strip():
            return str(session_id)
        channel = str(field_of(message, "channel", "default"))
        chat_id = str(field_of(message, "chat_id", "default"))
        return f"{channel}:{chat_id}"

    @hookimpl
    async def load_state(self, message: Message, session_id: str) -> TurnState:
        # SDK calls supply their agent before recovery so state comes from its store.
        agent = field_of(message, "_runtime_agent")
        if agent is None:
            agent = self._get_agent()
        state: TurnState = {"session_id": session_id, "_runtime_agent": agent}
        if context := field_of(message, "context_str"):
            state["context"] = context
        # Carry over a previously recorded per-session model override from the
        # session tape. Only set when a prior turn actually recorded one, so a
        # fresh/unknown session never inherits another session's model.
        if model := await self._recover_session_model(session_id, agent=agent):
            state["model"] = model
        if reasoning_effort := await self._recover_session_reasoning_effort(session_id, agent=agent):
            state["reasoning_effort"] = reasoning_effort
        if model := field_of(message, "context", {}).get("model"):
            state["model"] = model
        if thread_id := field_of(message, "context", {}).get("thread_id"):
            state["_runtime_thread_id"] = thread_id
        return state

    @hookimpl
    async def build_prompt(self, message: Message, session_id: str, state: TurnState) -> str | list[dict]:
        content = content_of(message)
        context = field_of(message, "context_str")
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        context_prefix = f"{context}\n---Date: {now}---\n" if context else ""
        text = f"{context_prefix}{content}"

        media = field_of(message, "media") or []
        if not media:
            return text

        media_parts: list[dict] = []
        for item in cast("list[MediaItem]", media):
            match item.type:
                case "image" | "video":
                    data_url = await item.get_url()
                    if not data_url:
                        continue
                    part_type = f"{item.type}_url"
                    media_parts.append({"type": part_type, part_type: {"url": data_url}})
                case "audio":
                    data_url = await item.get_url()
                    if data_url and (audio_part := _input_audio_part(data_url, item.mime_type)):
                        media_parts.append(audio_part)
                case _:
                    pass
        if media_parts:
            return [{"type": "text", "text": text}, *media_parts]
        return text

    @hookimpl
    async def run_model_stream(self, prompt: str | list[dict], session_id: str, state: TurnState) -> AsyncStreamEvents:
        return await self._get_agent(state).run_stream(
            session_id=session_id,
            prompt=prompt,
            state=state,
            model=state.get("model"),
        )

    @hookimpl
    def continue_prompt(self, prompt: str | list[dict], tape: Tape, state: StreamState) -> str:
        del prompt, state
        if "context" in tape.context.state:
            return f"{DEFAULT_CONTINUE_PROMPT} [context: {tape.context.state['context']}]"
        return DEFAULT_CONTINUE_PROMPT

    def _read_agents_file(self, state: TurnState) -> str:
        prompt_path = workspace_from_state(state) / AGENTS_FILE_NAME
        if not prompt_path.is_file():
            return ""
        try:
            return prompt_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    @hookimpl
    def system_prompt(self, prompt: str | list[dict], state: TurnState) -> str:
        # Read the content of AGENTS.md under workspace
        return DEFAULT_SYSTEM_PROMPT + "\n\n" + self._read_agents_file(state)

    @hookimpl
    def build_tape_context(self) -> TapeContext:
        return default_tape_context()

    @hookimpl
    async def before_tool_call(
        self,
        call: ToolCall,
        state: TurnState,
    ) -> ToolCallDecision | None:
        """Recover hallucinated/unknown tool names without interrupting the turn.

        When the model invokes a tool outside the current model-facing tool set,
        replace it with a guidance ``tool_result`` so the model can re-issue a
        valid call on the next step.
        """
        from bub.tools import model_tools

        agent = self._get_agent(state)

        available_tools = tuple(tool_item.name for tool_item in model_tools(agent.tools.values()))
        if call.tool in available_tools:
            return None

        matches = get_close_matches(call.tool, available_tools, n=3, cutoff=0.6)
        if matches:
            suggestions = "\n".join(f"- {name}" for name in matches)
            guidance = f"Tool `{call.tool}` does not exist. Did you mean one of the following?\n{suggestions}"
        elif "skill" in available_tools:
            guidance = f"Tool `{call.tool}` does not exist. Invoke the `skill` tool to list available skills."
        else:
            guidance = f"Tool `{call.tool}` does not exist. No similar tool is available."
        return ToolCallDecision.replace(guidance)


class BatteryImpl:
    """Optional builtin batteries: file tape store, tool-output spill, shell lifecycle

    Register this beside ``BuiltinImpl`` to opt into the batteries; the default
    builtin hooks keep the runtime free of them.
    """

    def __init__(self, framework: BubFramework) -> None:
        from bub.builtin.shell_manager import ShellManager

        self.framework = framework
        self.shell_manager = ShellManager()

    @hookimpl
    def load_state(self, message: Message, session_id: str) -> dict[str, Any]:
        del message, session_id
        from bub.builtin.tools import SHELL_MANAGER_KEY

        return {SHELL_MANAGER_KEY: self.shell_manager}

    @hookimpl
    def provide_tape_store(self) -> TapeStore:
        from bub.store import FileTapeStore

        return FileTapeStore(directory=self.framework.home / "tapes")

    @hookimpl
    def provide_tape_sidecar(self) -> TapeSidecar:
        from bub.builtin.spill import SpillSettings, SpillStore

        return SpillStore(self.framework.config.ensure(SpillSettings))

    @hookimpl
    async def provide_lifespan(self) -> AsyncIterator[None]:
        async with self.shell_manager.lifespan():
            yield

    @hookimpl(trylast=True)
    async def after_tool_call(
        self,
        call: ToolCall,
        result: ToolCallResult,
        state: TurnState,
    ) -> None:
        from bub.builtin.spill import SPILL_SIDECAR_NAME, SpillStore

        tape = state.get("_runtime_tape")
        if tape is None:
            return
        spill = tape.get_sidecar(SPILL_SIDECAR_NAME)
        if not isinstance(spill, SpillStore):
            return

        if result.error is None:
            tool_result = result.result
        elif isinstance(result.error, BubError):
            tool_result = result.error.as_dict() if result.result is None else result.result
        else:
            return

        rendered_result = render_tool_result(tool_result)
        bounded_result = await spill.spill_tool_result(
            tape,
            rendered_result,
            tool=call.tool,
            run_id=call.run_id,
        )
        if isinstance(tool_result, str) or bounded_result != rendered_result:
            result.result = bounded_result
