from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

CONFIG_MAP: dict[str, list[type[BaseSettings]]] = {}
ROOT = ""
MISSING = object()

_global_config: dict[str, list[BaseSettings]] = {}
_config_data: dict[str, Any] = {}


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


def load(config_file: Path) -> dict[str, Any]:
    """Load config from a file."""
    import yaml

    _global_config.clear()
    _config_data.clear()
    if config_file.exists():
        with config_file.open() as f:
            _config_data.update(yaml.safe_load(f) or {})
    return _config_data


def ensure_config[C: BaseSettings](config_cls: type[C]) -> C:
    """No-op function to ensure a config class is registered and can be imported."""
    section = getattr(config_cls, "__config_name__", ROOT)
    if section not in CONFIG_MAP:
        raise ValueError(f"No config registered for section '{section}'")

    instances = _global_config.setdefault(section, [])
    for instance in instances:
        if isinstance(instance, config_cls):
            return instance

    section_data = _config_data.get(section, {}) if section != ROOT else _config_data
    instance = config_cls.model_validate(section_data)
    instances.append(instance)
    return instance


def get_value(path: str, default: Any = MISSING) -> Any:
    """Get a loaded config value by dotted path, preserving registered settings behavior."""

    parts = [part for part in path.split(".") if part]
    if not parts:
        raise ValueError("config path must not be empty")

    value = _lookup_registered_config(parts)
    if value is not MISSING:
        return value

    if default is not MISSING:
        return default
    raise KeyError(path)


def _lookup_registered_config(parts: list[str]) -> Any:
    section, *subpath = parts
    if section in CONFIG_MAP and section != ROOT:
        for config_cls in CONFIG_MAP[section]:
            value = _lookup_path(ensure_config(config_cls), subpath)
            if value is not MISSING:
                return value

    for config_cls in CONFIG_MAP.get(ROOT, []):
        value = _lookup_path(ensure_config(config_cls), parts)
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
