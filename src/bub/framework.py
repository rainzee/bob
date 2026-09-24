"""Composable Bub framework runtime."""

from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast

import pluggy
from loguru import logger

from bub.configure import Config
from bub.envelope import Envelope, content_of, field_of
from bub.hooks.interception import AgentHooks
from bub.hooks.runtime import HookRuntime
from bub.hooks.specs import BUB_HOOK_NAMESPACE, BubHookSpecs
from bub.sidecars import TapeSidecar
from bub.store import AsyncTapeStore, TapeStore
from bub.streaming import StreamState
from bub.tape import Tape, TapeContext
from bub.turn import TurnResult, TurnState
from bub.utils import maybe_context_manager

DEFAULT_HOME = Path.home() / ".bub"


def resolve_home(home: Path | None) -> Path:
    """Resolve the Bub home directory from an explicit path, ``BUB_HOME``, or the user's home"""

    if home is not None:
        return home.expanduser().resolve()
    env_home = os.environ.get("BUB_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    return DEFAULT_HOME


class BubFramework:
    """Minimal framework core. Everything grows from hook skills."""

    def __init__(self, config_file: Path | None = None, home: Path | None = None) -> None:
        """Create a hook runtime and load this instance's configuration file.

        The workspace initially points to the current directory and the config file
        defaults to ``<home>/config.yml``. Register plugins or load builtin hooks
        before executing turns; construction does not load them.
        """
        self.workspace = Path.cwd().resolve()
        self.home = resolve_home(home)
        self.config_file = (config_file or self.home / "config.yml").resolve()
        self.config = Config()
        self._plugin_manager = pluggy.PluginManager(BUB_HOOK_NAMESPACE)
        self._plugin_manager.add_hookspecs(BubHookSpecs)
        self._hook_runtime = HookRuntime(self._plugin_manager)
        self._agent_hooks = AgentHooks(self._hook_runtime)
        self._tape_store: TapeStore | AsyncTapeStore | None = None
        self.config.load(self.config_file)

    @property
    def plugin_manager(self) -> pluggy.PluginManager:
        return self._plugin_manager

    def load_builtin_hooks(self, *, batteries: bool = False) -> None:
        """Load Bub's builtin hook implementations.

        Set ``batteries=True`` to also register the optional file tape store,
        tool-output spill, and shell lifecycle hooks.
        """
        from bub.builtin.hook_impl import BatteryImpl, BuiltinImpl

        try:
            self._plugin_manager.register(BuiltinImpl(self), name="builtin")
            if batteries:
                self._plugin_manager.register(BatteryImpl(self), name="batteries")
        except Exception as exc:
            logger.warning("Failed to load builtin hooks: {}", exc)

    def load_hooks(self, *, batteries: bool = False) -> None:
        """Load builtin hooks, then plugins from the ``bub`` entry-point group.

        Callable entry points receive this framework. A plugin that fails to load
        or initialize is logged and skipped so the remaining plugins still load.
        """
        import importlib.metadata

        pending_plugins: list[tuple[str, Any]] = []

        self.load_builtin_hooks(batteries=batteries)
        for entry_point in importlib.metadata.entry_points(group="bub"):
            try:
                plugin = entry_point.load()
            except Exception as exc:
                logger.warning(f"Failed to load plugin '{entry_point.name}': {exc}")
            else:
                pending_plugins.append((entry_point.name, plugin))

        for plugin_name, plugin in pending_plugins:
            try:
                if callable(plugin):  # Support entry points that are classes
                    plugin = plugin(self)
                self._plugin_manager.register(plugin, name=plugin_name)
            except Exception as exc:
                logger.warning(f"Failed to initialize plugin '{plugin_name}': {exc}")

    async def build_prompt(
        self, message: Envelope, session_id: str, state: dict[str, Any]
    ) -> str | list[dict[str, Any]]:
        """Build prompt for one message turn."""
        prompt = await self._hook_runtime.call_first(
            "build_prompt", message=message, session_id=session_id, state=state
        )
        if not prompt:
            prompt = content_of(message)
        return cast("str | list[dict[str, Any]]", prompt)

    async def continue_prompt(self, prompt: str | list[dict], tape: Tape, state: StreamState) -> str:
        """Build the prompt for the next step of an agent loop."""
        next_prompt = await self._hook_runtime.call_first("continue_prompt", prompt=prompt, tape=tape, state=state)
        if isinstance(next_prompt, str):
            return next_prompt
        raise TypeError("hook.continue_prompt must return str")

    async def build_state(self, message: Envelope, session_id: str) -> TurnState:
        """Merge runtime defaults and load-state hooks into a fresh turn state.

        Higher-priority hooks override lower-priority values. SDK callers can
        supply their Agent in the message's ``_runtime_agent`` field so builtin
        session recovery reads that agent's store.
        """
        state: dict[str, Any] = {"_runtime_workspace": str(self.workspace)}
        for hook_state in reversed(
            await self._hook_runtime.call_many("load_state", message=message, session_id=session_id)
        ):
            if isinstance(hook_state, dict):
                state.update(hook_state)
        return state

    async def process_inbound(self, inbound: Envelope) -> TurnResult:
        """Resolve, execute, and save one complete message turn."""

        try:
            session_id = await self.resolve_session(inbound)
            if isinstance(inbound, dict):
                inbound.setdefault("session_id", session_id)
            state = await self.build_state(inbound, session_id)
            prompt = await self.build_prompt(inbound, session_id, state)
            model_output = ""
            try:
                model_output = await self._run_model(inbound, prompt, session_id, state)
            finally:
                await self._hook_runtime.call_many(
                    "save_state",
                    session_id=session_id,
                    state=state,
                    message=inbound,
                    model_output=model_output,
                )

            return TurnResult(
                session_id=session_id,
                prompt=prompt,
                model_output=model_output,
                state=state,
            )
        except Exception as exc:
            logger.exception("Error processing inbound message")
            await self._hook_runtime.notify_error(stage="turn", error=exc, message=inbound)
            raise

    async def resolve_session(self, message: Envelope) -> str:
        """Resolve the canonical session id for a message."""

        resolved = await self._hook_runtime.call_first("resolve_session", message=message)
        return str(resolved or self._default_session_id(message))

    async def _run_model(
        self,
        inbound: Envelope,
        prompt: str | list[dict],
        session_id: str,
        state: dict[str, Any],
    ) -> str:
        output = await self._hook_runtime.run_model(prompt=prompt, session_id=session_id, state=state)
        if output is None:
            await self._hook_runtime.notify_error(
                stage="run_model",
                error=RuntimeError("no model skill returned output"),
                message=inbound,
            )
            return prompt if isinstance(prompt, str) else content_of(inbound)
        return output

    @staticmethod
    def _default_session_id(message: Envelope) -> str:
        session_id = field_of(message, "session_id")
        if session_id is not None:
            return str(session_id)
        channel = str(field_of(message, "channel", "default"))
        chat_id = str(field_of(message, "chat_id", "default"))
        return f"{channel}:{chat_id}"

    @contextlib.asynccontextmanager
    async def running(self) -> AsyncGenerator[contextlib.AsyncExitStack, None]:
        """Acquire hook-provided stores and resources for an application lifespan.

        Yield an AsyncExitStack for additional application resources. Exit closes
        acquired context managers and clears the framework's resource references.
        Enter before an Agent first accesses its cached tape; avoid overlapping
        lifespans on the same framework instance.
        """
        async with contextlib.AsyncExitStack() as stack:
            for lifespan in self._hook_runtime.call_many_sync("provide_lifespan"):
                await maybe_context_manager(lifespan, stack)
            tape_store = self._hook_runtime.call_first_sync("provide_tape_store")
            # Allow plugins to return either TapeStore/AsyncTapeStore instances or context managers for them
            # This benefits plugins that need to initialize and clean up resources with the tape store.
            self._tape_store = await maybe_context_manager(tape_store, stack)
            try:
                yield stack
            finally:
                self._tape_store = None

    def get_tape_store(self) -> TapeStore | AsyncTapeStore | None:
        """Return the store acquired by ``running()``, or None when unavailable."""
        return self._tape_store

    def get_tape_sidecars(self) -> tuple[TapeSidecar, ...]:
        """Collect tape sidecars, keeping the highest-priority provider for each name."""
        sidecars: dict[str, TapeSidecar] = {}
        for sidecar in self._hook_runtime.call_many_sync("provide_tape_sidecar"):
            sidecars.setdefault(sidecar.name, sidecar)
        return tuple(sidecars.values())

    def get_agent_hooks(self) -> AgentHooks:
        """Return the model and tool interception adapter for this framework's hooks."""
        return self._agent_hooks

    def get_system_prompt(self, prompt: str | list[dict], state: dict[str, Any]) -> str:
        """Join nonempty system-prompt hook results from low to high priority.

        Hooks contribute additional blocks; a higher-priority hook does not replace
        a lower-priority prompt. Blocks are separated by blank lines.
        """
        return "\n\n".join(
            result
            for result in reversed(self._hook_runtime.call_many_sync("system_prompt", prompt=prompt, state=state))
            if result
        )

    def build_tape_context(self) -> TapeContext:
        """Get the highest-priority tape context, raising TypeError if none is valid."""
        context = self._hook_runtime.call_first_sync("build_tape_context")
        if isinstance(context, TapeContext):
            return context
        raise TypeError("hook.build_tape_context must return TapeContext")
