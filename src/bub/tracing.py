"""Optional GenAI telemetry. Importing Bub never configures an exporter."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import opentelemetry.trace as otel
else:
    try:
        otel: Any = importlib.import_module("opentelemetry.trace")
    except ImportError:
        otel = None


def configure_otlp() -> None:
    """Configure opt-in HTTP export; preserve application-owned providers."""
    if (
        otel is None
        or not (os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
        or os.getenv("OTEL_SDK_DISABLED", "").lower() == "true"
        or not isinstance(otel.get_tracer_provider(), otel.ProxyTracerProvider)
    ):
        return
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return

    protocol = os.getenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL") or os.getenv(
        "OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"
    )
    if protocol != "http/protobuf":
        raise ValueError("Bub's trace extra supports OTLP http/protobuf only.")

    exporter = OTLPSpanExporter()
    # The SDK reads resource/sampler settings and drains the batch queue at process exit.
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    otel.set_tracer_provider(provider)


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError, RecursionError):
        return '"[unserializable value]"'


def _parts(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if message.get("role") == "tool":
        return [{"type": "tool_call_response", "id": message.get("tool_call_id", ""), "response": content}]
    parts: list[dict[str, Any]] = []
    if isinstance(content, str) and content:
        parts.append({"type": "text", "content": content})
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append({"type": "text", "content": part.get("text", "")})
            elif isinstance(part, dict):
                # Preserve media structure without copying inline binary payloads.
                parts.append({"type": "text", "content": f"[{part.get('type', 'media')} omitted]"})
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            with suppress(ValueError):
                arguments = json.loads(arguments)
        parts.append({
            "type": "tool_call",
            "id": call.get("id", ""),
            "name": function.get("name", ""),
            "arguments": arguments,
        })
    return parts


class Span:
    """A span whose lifetime is independent of its task-local activation."""

    def __init__(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        self._operation = str((attributes or {}).get("gen_ai.operation.name", ""))
        if otel is not None:
            self._span = otel.get_tracer("bub").start_span(
                name,
                kind=otel.SpanKind.CLIENT if self._operation == "chat" else otel.SpanKind.INTERNAL,
                attributes={k: v for k, v in (attributes or {}).items() if isinstance(v, str | bool | int | float)},
            )
        else:
            self._span = None
        self._ended = False
        self.set(**(attributes or {}))

    @property
    def recording(self) -> bool:
        return self._span is not None and self._span.is_recording()

    def set(self, **attributes: Any) -> None:
        if not self.recording:
            return
        for key, value in attributes.items():
            if value is not None:
                self._span.set_attribute(
                    key,
                    value
                    if isinstance(value, str | bool | int | float)
                    or (
                        key == "gen_ai.response.finish_reasons"
                        and isinstance(value, list | tuple)
                        and all(isinstance(v, str) for v in value)
                    )
                    else _json(value),
                )

    def rename(self, name: str) -> None:
        if self.recording:
            self._span.update_name(name)

    def messages(self, key: str, messages: list[dict[str, Any]]) -> None:
        if self.recording:
            normalized = [{"role": m.get("role", "user"), "parts": _parts(m)} for m in messages]
            self.set(**{key: _json(normalized)})
            self.set(**{
                key: _json(normalized),
            })

    def event(self, name: str, attributes: Mapping[str, Any]) -> None:
        if self.recording:
            self._span.add_event(
                name,
                {
                    k: v if isinstance(v, str | bool | int | float) else _json(v)
                    for k, v in attributes.items()
                    if v is not None
                },
            )

    def fail(self, error: BaseException) -> None:
        if not self.recording:
            return
        if isinstance(error, asyncio.CancelledError | GeneratorExit):
            self.set(**{"bub.cancelled": True})
        else:
            self._span.record_exception(error)
            self._span.set_status(otel.Status(otel.StatusCode.ERROR, str(error)))
            self.set(**{"error.type": type(error).__name__})

    def end(self) -> None:
        if not self._ended:
            self._ended = True
            if self._span is not None:
                self._span.end()

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _CURRENT.set(self)
        try:
            if self._span is None:
                yield
            else:
                with otel.use_span(
                    self._span, end_on_exit=False, record_exception=False, set_status_on_exception=False
                ):
                    yield
        finally:
            _CURRENT.reset(token)

    def correlation(self) -> dict[str, str]:
        if self._span is None:
            return {}
        context = self._span.get_span_context()
        if not context.is_valid:
            return {}
        return {"trace_id": f"{context.trace_id:032x}", "span_id": f"{context.span_id:016x}"}


_CURRENT: ContextVar[Span | None] = ContextVar("bub_trace_span", default=None)


def current_span() -> Span | None:
    return _CURRENT.get()


def correlation() -> dict[str, str]:
    span = current_span()
    return span.correlation() if span else {}


def event(event_name: str, **attributes: Any) -> None:
    if span := current_span():
        span.event(event_name, attributes)
