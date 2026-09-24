"""Generic Pluggy execution with per-adapter fault isolation."""

from __future__ import annotations

import inspect
from typing import Any

import pluggy
from loguru import logger


class HookRuntime:
    """Safe wrapper around Pluggy hook execution."""

    def __init__(self, plugin_manager: pluggy.PluginManager) -> None:
        self._plugin_manager = plugin_manager

    async def call_first(self, hook_name: str, **kwargs: Any) -> Any:
        """Run hook implementations in precedence order and return first non-None value."""

        for impl in self._iter_hookimpls(hook_name):
            call_kwargs = self._kwargs_for_impl(impl, kwargs)
            value = await self._invoke_impl_async(
                hook_name=hook_name, impl=impl, call_kwargs=call_kwargs, kwargs=kwargs
            )
            if value is _SKIP_VALUE:
                continue
            if value is not None:
                return value
        return None

    async def call_many(self, hook_name: str, **kwargs: Any) -> list[Any]:
        """Run all implementations and collect successful return values."""

        results: list[Any] = []
        for impl in self._iter_hookimpls(hook_name):
            call_kwargs = self._kwargs_for_impl(impl, kwargs)
            value = await self._invoke_impl_async(
                hook_name=hook_name, impl=impl, call_kwargs=call_kwargs, kwargs=kwargs
            )
            if value is _SKIP_VALUE:
                continue
            results.append(value)
        return results

    def call_first_sync(self, hook_name: str, **kwargs: Any) -> Any:
        """Synchronous variant of ``call_first`` for bootstrap hooks."""

        for impl in self._iter_hookimpls(hook_name):
            call_kwargs = self._kwargs_for_impl(impl, kwargs)
            value = self._invoke_impl_sync(hook_name=hook_name, impl=impl, call_kwargs=call_kwargs, kwargs=kwargs)
            if value is _SKIP_VALUE:
                continue
            if value is not None:
                return value
        return None

    def call_many_sync(self, hook_name: str, **kwargs: Any) -> list[Any]:
        """Synchronous variant of ``call_many`` for bootstrap hooks."""

        results: list[Any] = []
        for impl in self._iter_hookimpls(hook_name):
            call_kwargs = self._kwargs_for_impl(impl, kwargs)
            value = self._invoke_impl_sync(hook_name=hook_name, impl=impl, call_kwargs=call_kwargs, kwargs=kwargs)
            if value is _SKIP_VALUE:
                continue
            results.append(value)
        return results

    async def _invoke_impl_async(
        self,
        *,
        hook_name: str,
        impl: Any,
        call_kwargs: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> Any:
        value = impl.function(**call_kwargs)
        if inspect.isawaitable(value):
            value = await value
        return value

    def _invoke_impl_sync(
        self,
        *,
        hook_name: str,
        impl: Any,
        call_kwargs: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> Any:
        value = impl.function(**call_kwargs)
        if inspect.isawaitable(value):
            logger.warning(
                "hook.async_not_supported hook={} adapter={}",
                hook_name,
                impl.plugin_name or "<unknown>",
            )
            return _SKIP_VALUE
        return value

    def _iter_hookimpls(self, hook_name: str) -> list[Any]:
        hook = getattr(self._plugin_manager.hook, hook_name, None)
        if hook is None or not hasattr(hook, "get_hookimpls"):
            return []
        return list(reversed(hook.get_hookimpls()))

    @staticmethod
    def _kwargs_for_impl(impl: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {name: kwargs[name] for name in impl.argnames if name in kwargs}


_SKIP_VALUE = object()
