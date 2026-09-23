from collections.abc import AsyncIterator, Iterator
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

from bub.tape import TapeEntry
from bub.turn import TurnState


def workspace_from_state(state: TurnState) -> Path:
    raw = state.get("_runtime_workspace")
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser().resolve()
    return Path.cwd().resolve()


def get_entry_text(entry: TapeEntry) -> str:
    import yaml

    return yaml.safe_dump(entry.payload)


async def maybe_context_manager(obj: Any, stack: AsyncExitStack) -> Any:
    """Enter the context manager if the obj is any kind of iterator, otherwise return the obj as is."""
    if isinstance(obj, AsyncIterator):
        obj = await stack.enter_async_context(asynccontextmanager(lambda: obj)())
    elif isinstance(obj, Iterator):
        obj = stack.enter_context(contextmanager(lambda: obj)())
    return obj
