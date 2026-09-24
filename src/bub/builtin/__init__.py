from .agent import Agent
from .tools import TOOLS


def battery_tools() -> tuple:
    """Builtin tool set: shells, filesystem, tape, web fetch, subagent, spill reader"""

    from bub.builtin.spill import spill_read

    return (*TOOLS, spill_read)


__all__ = ["Agent", "battery_tools"]
