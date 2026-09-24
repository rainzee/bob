"""Composable Bub framework runtime."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path

from bub.hooks import Hooks
from bub.sidecars import TapeSidecar
from bub.store import AsyncTapeStore, TapeStore
from bub.turn import TurnState
from bub.utils import LifespanFactory, maybe_context_manager


class BubFramework:
    """Composition root: explicit paths, resources and callbacks

    The host assembles the runtime by hand -- there is no registry, no discovery and
    no name lookup. Every contribution goes through one of the named slots below.
    """

    def __init__(self, *, workspace: Path, home: Path) -> None:
        """Create a runtime from explicitly configured paths.

        Args:
            workspace: Directory turns and skill discovery resolve against.
            home: Directory the runtime may write tapes under.
        """
        self.workspace = workspace.expanduser().resolve()
        self.home = home.expanduser().resolve()
        self.hooks = Hooks()
        self._tape_store: TapeStore | AsyncTapeStore | None = None
        self._active_tape_store: TapeStore | AsyncTapeStore | None = None
        self._sidecars: dict[str, TapeSidecar] = {}
        self._lifespans: list[LifespanFactory] = []

    def add_hooks(self, hooks: Hooks) -> None:
        """Append one set of callbacks; later callbacks run and win over earlier ones."""

        self.hooks = self.hooks + hooks

    def add_tape_store(self, store: TapeStore | AsyncTapeStore | None) -> None:
        """Declare the default store turns use; it is entered by ``running()``.

        The value may be a store, or an iterator yielding the store when the store
        owns resources that need entering and exiting around a lifespan.
        """

        self._tape_store = store

    def add_sidecars(self, *sidecars: TapeSidecar) -> None:
        """Mount sidecars; a later sidecar with the same name replaces an earlier one."""

        for sidecar in sidecars:
            self._sidecars[sidecar.name] = sidecar

    def add_lifespans(self, *lifespans: LifespanFactory) -> None:
        """Register factories started by ``running()`` and stopped when it exits."""

        self._lifespans.extend(lifespans)

    @contextlib.asynccontextmanager
    async def running(self) -> AsyncGenerator[contextlib.AsyncExitStack, None]:
        """Acquire the registered lifespans and the default store.

        Yield an AsyncExitStack for additional application resources. Exit closes
        acquired context managers and clears the runtime's active store.
        Enter before an Agent first accesses its cached tape; avoid overlapping
        lifespans on the same framework instance.
        """
        async with contextlib.AsyncExitStack() as stack:
            for lifespan in self._lifespans:
                await maybe_context_manager(lifespan(), stack)
            self._active_tape_store = await maybe_context_manager(self._tape_store, stack)
            try:
                yield stack
            finally:
                self._active_tape_store = None

    def get_tape_store(self) -> TapeStore | AsyncTapeStore | None:
        """Return the store acquired by ``running()``, or None when unavailable."""

        return self._active_tape_store

    def get_tape_sidecars(self) -> tuple[TapeSidecar, ...]:
        """Return the mounted sidecars in first-seen order."""

        return tuple(self._sidecars.values())

    async def build_state(self, session_id: str, state: TurnState | None = None) -> TurnState:
        """Resolve one session's turn state from defaults, seeds, and load-state callbacks.

        Args:
            session_id: Session identity within the workspace.
            state: Values the caller already knows, for example ``_runtime_agent``.

        Callbacks receive the accumulated state and may return a partial update;
        later callbacks override earlier ones.
        """
        resolved: TurnState = {"_runtime_workspace": str(self.workspace), **(state or {})}
        return await self.hooks.run_load_state(session_id, resolved)
