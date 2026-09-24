from __future__ import annotations

from bub.builtin.agent import Agent
from bub.builtin.hooks import Battery, BuiltinHooks
from bub.builtin.tools import TOOLS


def battery_tools() -> tuple:
    """Builtin tool set: shells, filesystem, tape, web fetch, subagent, spill reader"""

    from bub.builtin.spill import spill_read

    return (*TOOLS, spill_read)


__all__ = ["Agent", "Battery", "BuiltinHooks", "battery_tools"]
