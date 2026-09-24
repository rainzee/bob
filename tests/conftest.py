from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from bub.model_runner import ChatRequest


def text_chunk(text: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


def reasoning_chunk(text: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"reasoning": text}}]}


def tool_calls_chunk(calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, **c} for i, c in enumerate(calls)]}}]}


def usage_chunk(usage: dict[str, Any]) -> dict[str, Any]:
    return {"choices": [], "usage": usage}


def call(name: str, arguments: str = "{}") -> dict[str, Any]:
    return {"id": f"call-{name}", "type": "function", "function": {"name": name, "arguments": arguments}}


class RecordingClient:
    """测试用的 chat client: 按脚本回放 chunk, 并记录收到的每个 ChatRequest"""

    def __init__(self, *scripts: list[dict[str, Any]]) -> None:
        self.requests: list[ChatRequest] = []
        self.closed = 0
        self._scripts = list(scripts)

    def stream(self, request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        self.requests.append(request)
        script = self._scripts.pop(0) if self._scripts else [text_chunk("done")]

        async def chunks() -> AsyncIterator[dict[str, Any]]:
            try:
                for chunk in script:
                    yield chunk
            finally:
                self.closed += 1

        return chunks()
