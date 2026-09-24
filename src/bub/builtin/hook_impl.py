from collections.abc import AsyncIterator
from difflib import get_close_matches
from typing import Any, cast

from bub.builtin.agent import Agent
from bub.builtin.context import default_tape_context, render_tool_result
from bub.errors import BubError
from bub.framework import BubFramework
from bub.hooks import hookimpl
from bub.hooks.interception import ToolCall, ToolCallDecision, ToolCallResult
from bub.sidecars import TapeSidecar
from bub.store import TapeStore
from bub.streaming import StreamState
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
    async def load_state(self, session_id: str, state: TurnState) -> TurnState:
        agent = state.get("_runtime_agent")
        if not isinstance(agent, Agent):
            agent = self._get_agent()
        loaded: TurnState = {"session_id": session_id, "_runtime_agent": agent}
        # Carry over a previously recorded per-session model override from the
        # session tape. Only set when a prior turn actually recorded one, so a
        # fresh/unknown session never inherits another session's model.
        if model := await self._recover_session_model(session_id, agent=agent):
            loaded["model"] = model
        if reasoning_effort := await self._recover_session_reasoning_effort(session_id, agent=agent):
            loaded["reasoning_effort"] = reasoning_effort
        return loaded

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
    def load_state(self, session_id: str, state: TurnState) -> dict[str, Any]:
        del session_id, state
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
