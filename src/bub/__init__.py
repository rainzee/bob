"""Bub: a tiny agent runtime, embedded as a library"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version

from bub.agent import Agent
from bub.errors import BubError, ErrorKind
from bub.framework import BubFramework
from bub.hooks import (
    Hooks,
    LlmCallDecision,
    LlmCallRequest,
    LlmCallResult,
    ToolCall,
    ToolCallDecision,
    ToolCallResult,
)
from bub.model_runner import ChatClient, ChatRequest
from bub.sidecars import TapeSidecar
from bub.store import AsyncTapeStore, FileTapeStore, InMemoryTapeStore, TapeStore
from bub.streaming import AsyncStreamEvents, StreamEvent
from bub.tape import Tape, TapeContext, TapeEntry
from bub.tools import Tool, ToolContext, tool
from bub.turn import TurnState

__all__ = [
    "Agent",
    "AsyncStreamEvents",
    "AsyncTapeStore",
    "BubError",
    "BubFramework",
    "ChatClient",
    "ChatRequest",
    "ErrorKind",
    "FileTapeStore",
    "Hooks",
    "InMemoryTapeStore",
    "LlmCallDecision",
    "LlmCallRequest",
    "LlmCallResult",
    "StreamEvent",
    "Tape",
    "TapeContext",
    "TapeEntry",
    "TapeSidecar",
    "TapeStore",
    "Tool",
    "ToolCall",
    "ToolCallDecision",
    "ToolCallResult",
    "ToolContext",
    "TurnState",
    "tool",
]

try:
    __version__ = import_module("bub._version").version
except ModuleNotFoundError:
    try:
        __version__ = metadata_version("bub")
    except PackageNotFoundError:
        __version__ = "0.0.0"
