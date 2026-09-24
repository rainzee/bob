from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from any_llm import AnyLLM
from any_llm.constants import LLMProvider
from pydantic import ConfigDict, Field, field_validator

from bub.configure import Settings, config

DEFAULT_MAX_TOKENS = 16384


@dataclass(frozen=True)
class ModelCandidate:
    provider: LLMProvider
    model_id: str
    name: str


@config()
class AgentSettings(Settings):
    """Configuration settings for the Agent."""

    model_config = ConfigDict(extra="ignore")
    model: str
    fallback_models: list[str] | None = None
    api_key: str | dict[str, str] | None = None
    api_base: str | dict[str, str] | None = None
    max_steps: int = Field(default=sys.maxsize, gt=0)
    max_tokens: int = DEFAULT_MAX_TOKENS
    model_timeout_seconds: int | None = None
    client_args: dict[str, Any] = Field(default_factory=dict)
    completion_args: dict[str, Any] = Field(default_factory=dict)
    verbose: int = Field(default=0, description="Verbosity level for logging. Higher means more verbose.", ge=0, le=2)

    @field_validator("client_args", "completion_args", mode="before")
    @classmethod
    def default_dict_args(cls, value: Any) -> Any:
        return {} if value is None else value

    def model_candidates(self, model: str) -> list[ModelCandidate]:
        candidate_names = [model]
        if model == self.model:
            candidate_names.extend(self.fallback_models or [])

        candidates: list[ModelCandidate] = []
        for candidate in candidate_names:
            provider, model_id = AnyLLM.split_model_provider(candidate)
            candidates.append(ModelCandidate(provider=provider, model_id=model_id, name=candidate))
        return candidates

    def model_client_kwargs(self, provider: str) -> dict[str, Any]:
        return {
            **self.client_args,
            "api_key": self._provider_value(self.api_key, provider),
            "api_base": self._provider_value(self.api_base, provider),
        }

    @staticmethod
    def _provider_value(value: str | dict[str, str] | None, provider: str) -> str | None:
        if isinstance(value, dict):
            return value.get(provider)
        return value
