from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    AsyncExitStack,
    asynccontextmanager,
    contextmanager,
)
from pathlib import Path
from typing import Any

from bub.turn import TurnState

type Lifespan = AsyncIterator[None] | AbstractAsyncContextManager[None] | Iterator[None] | AbstractContextManager[None]
type LifespanFactory = Callable[[], Lifespan]
type MaybeAwait[T] = T | Awaitable[T]


def workspace_from_state(state: TurnState) -> Path:
    """Return the workspace recorded on the turn state; the framework always sets it"""

    raw = state.get("_runtime_workspace")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("turn state has no _runtime_workspace; pass state built by the framework")
    return Path(raw).expanduser().resolve()


async def maybe_context_manager(obj: Any, stack: AsyncExitStack) -> Any:
    """Enter any flavour of context manager or generator; return anything else as is."""

    if hasattr(obj, "__aenter__"):
        return await stack.enter_async_context(obj)
    if hasattr(obj, "__enter__"):
        return stack.enter_context(obj)
    if isinstance(obj, AsyncIterator):
        return await stack.enter_async_context(asynccontextmanager(lambda: obj)())
    if isinstance(obj, Iterator):
        return stack.enter_context(contextmanager(lambda: obj)())
    return obj
