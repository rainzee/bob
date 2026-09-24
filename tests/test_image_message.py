"""Tests for image/media message handling through the pipeline."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from bub.builtin.hook_impl import BuiltinImpl
from bub.framework import BubFramework
from bub.message import MediaItem, Message


def _async_return(value):
    async def runner(*args, **kwargs):
        return value

    return runner


# ---------------------------------------------------------------------------
# MediaItem & Message
# ---------------------------------------------------------------------------


def test_media_item_keeps_fetcher_and_filename() -> None:
    async def fetch_bytes() -> bytes:
        return b"abc"

    item = MediaItem(type="image", mime_type="image/jpeg", filename="a.jpg", data_fetcher=fetch_bytes)

    assert item.type == "image"
    assert item.mime_type == "image/jpeg"
    assert item.filename == "a.jpg"
    assert item.data_fetcher is fetch_bytes


@pytest.mark.asyncio
async def test_media_item_returns_none_when_fetcher_skips_download() -> None:
    item = MediaItem(type="video", mime_type="video/mp4", data_fetcher=_async_return(None))

    assert await item.get_url() is None


def test_message_defaults_keep_text_only_messages_media_free() -> None:
    message = Message(session_id="s", channel="tg", content="hello")

    assert message.media == []
    assert message.context_str == ""


# ---------------------------------------------------------------------------
# build_prompt with media
# ---------------------------------------------------------------------------


class FakeAgent:
    def __init__(self, home: Path) -> None:
        self.settings = SimpleNamespace(home=home)


def _build_impl(tmp_path: Path) -> tuple[BubFramework, BuiltinImpl]:
    framework = BubFramework(workspace=tmp_path, home=tmp_path)
    impl = BuiltinImpl(framework)
    impl._agent = FakeAgent(tmp_path)
    return framework, impl


@pytest.mark.asyncio
async def test_build_prompt_returns_string_without_media(tmp_path: Path) -> None:
    _, impl = _build_impl(tmp_path)
    message = Message(session_id="s", channel="tg", content="hello")

    result = await impl.build_prompt(message, session_id="s", state={})

    assert isinstance(result, str)
    assert "hello" in result


@pytest.mark.asyncio
async def test_build_prompt_returns_multimodal_parts_with_image_media(tmp_path: Path) -> None:
    _, impl = _build_impl(tmp_path)
    message = Message(
        session_id="s",
        channel="tg",
        content="describe this",
        media=[MediaItem(type="image", mime_type="image/jpeg", data_fetcher=_async_return(b"\xff\xd8"))],
    )

    result = await impl.build_prompt(message, session_id="s", state={})

    assert isinstance(result, list)
    assert len(result) == 2

    text_part = result[0]
    assert text_part["type"] == "text"
    assert "describe this" in text_part["text"]

    image_part = result[1]
    assert image_part["type"] == "image_url"
    expected = base64.b64encode(b"\xff\xd8").decode("utf-8")
    assert image_part["image_url"]["url"] == f"data:image/jpeg;base64,{expected}"


@pytest.mark.asyncio
async def test_build_prompt_with_multiple_images(tmp_path: Path) -> None:
    _, impl = _build_impl(tmp_path)
    message = Message(
        session_id="s",
        channel="tg",
        content="compare these",
        media=[
            MediaItem(type="image", mime_type="image/jpeg", data_fetcher=_async_return(b"A")),
            MediaItem(type="image", mime_type="image/jpeg", data_fetcher=_async_return(b"B")),
        ],
    )

    result = await impl.build_prompt(message, session_id="s", state={})

    assert isinstance(result, list)
    assert len(result) == 3
    assert result[1]["type"] == "image_url"
    assert result[2]["type"] == "image_url"


@pytest.mark.asyncio
async def test_build_prompt_returns_video_url_part_with_video_media(tmp_path: Path) -> None:
    _, impl = _build_impl(tmp_path)
    message = Message(
        session_id="s",
        channel="tg",
        content="describe this video",
        media=[MediaItem(type="video", mime_type="video/mp4", data_fetcher=_async_return(b"video"))],
    )

    result = await impl.build_prompt(message, session_id="s", state={})

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["type"] == "text"
    assert "describe this video" in result[0]["text"]
    expected = base64.b64encode(b"video").decode("utf-8")
    assert result[1] == {
        "type": "video_url",
        "video_url": {"url": f"data:video/mp4;base64,{expected}"},
    }


@pytest.mark.asyncio
async def test_build_prompt_skips_video_when_download_is_too_large(tmp_path: Path) -> None:
    _, impl = _build_impl(tmp_path)
    message = Message(
        session_id="s",
        channel="tg",
        content="describe this video",
        media=[MediaItem(type="video", mime_type="video/mp4", data_fetcher=_async_return(None))],
    )

    result = await impl.build_prompt(message, session_id="s", state={})

    assert isinstance(result, str)
    assert "describe this video" in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mime_type", "expected_format"),
    [("audio/mpeg", "mp3"), ("audio/ogg", "ogg"), ("audio/x-wav", "wav")],
)
async def test_build_prompt_returns_input_audio_part(tmp_path: Path, mime_type: str, expected_format: str) -> None:
    _, impl = _build_impl(tmp_path)
    message = Message(
        session_id="s",
        channel="tg",
        content="listen to this",
        media=[MediaItem(type="audio", mime_type=mime_type, data_fetcher=_async_return(b"audio"))],
    )

    result = await impl.build_prompt(message, session_id="s", state={})

    assert isinstance(result, list)
    assert result[0]["type"] == "text"
    assert "listen to this" in result[0]["text"]
    assert result[1] == {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(b"audio").decode("utf-8"),
            "format": expected_format,
        },
    }


@pytest.mark.asyncio
async def test_build_prompt_skips_remote_audio_url(tmp_path: Path) -> None:
    _, impl = _build_impl(tmp_path)
    message = Message(
        session_id="s",
        channel="tg",
        content="listen to this",
        media=[MediaItem(type="audio", mime_type="audio/ogg", url="https://example.com/audio.ogg")],
    )

    result = await impl.build_prompt(message, session_id="s", state={})

    assert isinstance(result, str)
    assert "listen to this" in result


@pytest.mark.asyncio
async def test_build_prompt_keeps_media_for_text_starting_with_comma(tmp_path: Path) -> None:
    _, impl = _build_impl(tmp_path)
    message = Message(
        session_id="s",
        channel="tg",
        content=",help",
        media=[MediaItem(type="image", mime_type="image/jpeg", data_fetcher=_async_return(b"X"))],
    )

    result = await impl.build_prompt(message, session_id="s", state={})

    assert isinstance(result, list)
    assert result[0]["type"] == "text"
    assert result[0]["text"].endswith(",help")


# ---------------------------------------------------------------------------
# _extract_text_from_parts
# ---------------------------------------------------------------------------


def test_extract_text_from_parts() -> None:
    from bub.builtin.agent import _extract_text_from_parts

    parts = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,X"}},
        {"type": "text", "text": "world"},
    ]
    assert _extract_text_from_parts(parts) == "hello\nworld"


def test_extract_text_from_parts_empty() -> None:
    from bub.builtin.agent import _extract_text_from_parts

    assert _extract_text_from_parts([]) == ""


def test_extract_text_from_parts_no_text_parts() -> None:
    from bub.builtin.agent import _extract_text_from_parts

    parts = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,X"}}]
    assert _extract_text_from_parts(parts) == ""
