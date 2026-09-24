from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

CONFIG_MAP: dict[str, list[type[BaseModel]]] = {}
ROOT = ""
MISSING = object()


class Settings(BaseModel):
    """Base for plugin settings; values come only from explicit configuration"""

    model_config = ConfigDict(extra="ignore")


def config[C: type[BaseModel]](name: str = ROOT) -> Callable[[C], C]:
    """Decorator to register a config class for a plugin."""

    def decorator(cls: C) -> C:
        cls.__config_name__ = name  # type: ignore[attr-defined]
        if name not in CONFIG_MAP:
            CONFIG_MAP[name] = []
        CONFIG_MAP[name].append(cls)
        return cls

    return decorator


class Config:
    """Explicit configuration for one framework instance"""

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(data or {})
        self._instances: dict[str, list[BaseModel]] = {}

    @classmethod
    def from_file(cls, path: Path) -> Config:
        """Read configuration from a YAML file; a missing file yields empty configuration"""

        import yaml

        if not path.is_file():
            return cls()
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    @property
    def data(self) -> dict[str, Any]:
        return dict(self._data)

    def ensure[C: BaseModel](self, config_cls: type[C]) -> C:
        """Return the cached instance of a registered config class, or build it"""

        section = getattr(config_cls, "__config_name__", ROOT)
        if section not in CONFIG_MAP:
            raise ValueError(f"No config registered for section '{section}'")

        instances = self._instances.setdefault(section, [])
        for instance in instances:
            if isinstance(instance, config_cls):
                return instance

        section_data = self._data.get(section, {}) if section != ROOT else self._data
        instance = config_cls.model_validate(section_data)
        instances.append(instance)
        return instance

    def get_value(self, path: str, default: Any = MISSING) -> Any:
        """Get a loaded config value by dotted path, preserving registered settings behavior."""

        parts = [part for part in path.split(".") if part]
        if not parts:
            raise ValueError("config path must not be empty")

        value = self._lookup_registered_config(parts)
        if value is not MISSING:
            return value

        if default is not MISSING:
            return default
        raise KeyError(path)

    def _lookup_registered_config(self, parts: list[str]) -> Any:
        section, *subpath = parts
        if section in CONFIG_MAP and section != ROOT:
            for config_cls in CONFIG_MAP[section]:
                value = _lookup_path(self.ensure(config_cls), subpath)
                if value is not MISSING:
                    return value

        for config_cls in CONFIG_MAP.get(ROOT, []):
            value = _lookup_path(self.ensure(config_cls), parts)
            if value is not MISSING:
                return value

        return MISSING


def _lookup_path(value: Any, parts: list[str]) -> Any:
    current = value
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                return MISSING
            current = current[part]
            continue
        if not hasattr(current, part):
            return MISSING
        current = getattr(current, part)
    return current
