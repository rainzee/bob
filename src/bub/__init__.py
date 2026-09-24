"""Bub framework package."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version

from bub.builtin.agent import Agent
from bub.configure import Config, Settings
from bub.framework import BubFramework
from bub.hooks import Hooks
from bub.tools import Tool, ToolContext, tool

__all__ = [
    "Agent",
    "BubFramework",
    "Config",
    "Hooks",
    "Settings",
    "Tool",
    "ToolContext",
    "tool",
]

try:
    __version__ = import_module("bub._version").version
except ModuleNotFoundError:
    try:
        __version__ = metadata_version("bub")
    except PackageNotFoundError:
        __version__ = "0.0.0"
