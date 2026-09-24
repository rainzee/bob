from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict

ROOT = ""
MISSING = object()


class Settings(BaseModel):
    """配置段的基础类, 取值只来自显式传入的映射"""

    model_config = ConfigDict(extra="ignore")
    section: ClassVar[str] = ROOT


class Config:
    """一个运行时实例持有的显式配置"""

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(data or {})
        self._instances: dict[type[BaseModel], BaseModel] = {}

    @classmethod
    def from_file(cls, path: Path) -> Config:
        """从 YAML 文件读取配置; 文件不存在时得到空配置"""

        import yaml

        if not path.is_file():
            return cls()
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    @property
    def data(self) -> dict[str, Any]:
        return dict(self._data)

    def get_value(self, path: str, default: Any = MISSING) -> Any:
        """按点分路径读取原始配置值, 没有默认值时缺失的路径抛 KeyError"""

        parts = [part for part in path.split(".") if part]
        if not parts:
            raise ValueError("config path must not be empty")

        value = _lookup_path(self._data, parts)
        if value is not MISSING:
            return value
        if default is not MISSING:
            return default
        raise KeyError(path)

    def ensure[C: Settings](self, settings_cls: type[C]) -> C:
        """按类上声明的 section 实例化配置, 同一个 Config 内按类缓存"""

        cached = self._instances.get(settings_cls)
        if cached is not None:
            return cast("C", cached)

        section = settings_cls.section
        raw = self._data if not section else (self._data.get(section) or {})
        instance = settings_cls.model_validate(raw)
        self._instances[settings_cls] = instance
        return instance


def _lookup_path(data: Mapping[str, Any], parts: list[str]) -> Any:
    current: Any = data
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            return MISSING
        current = current[part]
    return current
