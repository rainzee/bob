from __future__ import annotations

import base64
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

type MediaType = Literal["image", "audio", "video", "document"]

_AUDIO_FORMAT_TO_MIME_TYPE = {
    "aiff": "audio/aiff",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "wav": "audio/wav",
    "webm": "audio/webm",
}
_AUDIO_MIME_TYPE_TO_FORMAT = {
    **{mime_type: audio_format for audio_format, mime_type in _AUDIO_FORMAT_TO_MIME_TYPE.items()},
    "audio/x-aiff": "aiff",
    "audio/x-flac": "flac",
    "audio/x-m4a": "m4a",
    "audio/x-wav": "wav",
}


def audio_format_from_mime_type(mime_type: str) -> str:
    normalized = mime_type.partition(";")[0].strip().lower()
    return _AUDIO_MIME_TYPE_TO_FORMAT.get(normalized, normalized.removeprefix("audio/") or "unknown")


def audio_mime_type_from_format(audio_format: str) -> str:
    return _AUDIO_FORMAT_TO_MIME_TYPE.get(audio_format, f"audio/{audio_format}")


@dataclass
class MediaItem:
    """An attachment carried by an inbound message"""

    type: MediaType
    mime_type: str
    filename: str | None = None
    url: str | None = None
    data_fetcher: Callable[[], Awaitable[bytes | None]] | None = None

    async def get_url(self) -> str | None:
        """Get a URL for the media, fetching data if necessary"""

        if self.url:
            return self.url
        if self.data_fetcher is not None:
            data = await self.data_fetcher()
            if data is None:
                return None
            return f"data:{self.mime_type};base64,{base64.b64encode(data).decode('utf-8')}"
        return None


@dataclass
class Message:
    """Concrete inbound envelope a host can pass to the framework"""

    session_id: str
    content: str
    channel: str = "default"
    chat_id: str = "default"
    context: dict[str, Any] = field(default_factory=dict)
    media: list[MediaItem] = field(default_factory=list)

    @property
    def context_str(self) -> str:
        """Render the context for prompt building"""

        return "|".join(f"{key}={value}" for key, value in self.context.items() if not key.startswith("_"))
