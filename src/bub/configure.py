from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

CONFIG_MAP: dict[str, list[type[BaseSettings]]] = {}
ROOT = ""
MISSING = object()


class Settings(BaseSettings):
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del settings_cls  # unused
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)


def config[C: type[BaseSettings]](name: str = ROOT) -> Callable[[C], C]:
    """Decorator to register a config class for a plugin."""

    def decorator(cls: C) -> C:
        cls.__config_name__ = name  # type: ignore[attr-defined]
        if name not in CONFIG_MAP:
            CONFIG_MAP[name] = []
        CONFIG_MAP[name].append(cls)
        return cls

    return decorator


class Config:
    """One parsed configuration file, owned by a framework instance"""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._instances: dict[str, list[BaseSettings]] = {}

    def load(self, config_file: Path) -> dict[str, Any]:
        """Replace the loaded data with the contents of ``config_file``"""

        import yaml

        self._instances.clear()
        self._data.clear()
        if config_file.exists():
            with config_file.open() as f:
                self._data.update(yaml.safe_load(f) or {})
        return self._data

    def ensure[C: BaseSettings](self, config_cls: type[C]) -> C:
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
