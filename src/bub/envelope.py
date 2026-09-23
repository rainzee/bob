"""Utilities for reading and normalizing user-defined envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

type Envelope = Any


def field_of(message: Envelope, key: str, default: Any = None) -> Any:
    """Read a field from mapping-like or attribute-based messages."""

    if isinstance(message, Mapping):
        return message.get(key, default)
    return getattr(message, key, default)


def content_of(message: Envelope) -> str:
    """Get textual content from any envelope shape."""

    return str(field_of(message, "content", ""))
