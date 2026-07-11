"""The YAML model-config file shape (``configs/models/*.yaml``) and its conversion into the
runtime types the engine actually uses (:class:`ModelSpec`, :class:`RunConfig`).

Only ``provider``/``model``/``generation``/``runtime`` are meaningful today (the only registered
provider is ``mock``, which needs no ``base_url``/``api_key_env``); those two fields are already
part of the schema so real-provider config files (Milestone 5) don't need a breaking format
change, they just start being read.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from financebench.schemas.common import (
    DEFAULT_PROMPT_PROFILE,
    ConversationProtocol,
    EvalMode,
)
from financebench.schemas.model_io import ModelSpec
from financebench.schemas.run import CacheMode, RunConfig
from financebench.utils.errors import ConfigError

__all__ = ["GenerationConfig", "ModelConfigFile", "RuntimeConfig", "load_model_config"]


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float = 0.0
    max_output_tokens: int = 1024
    timeout_seconds: float = 120.0


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    concurrency: int = 4
    retries: int = 4
    cache: bool = True


class ModelConfigFile(BaseModel):
    """The validated contents of a ``configs/models/*.yaml`` file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    def to_model_spec(self) -> ModelSpec:
        return ModelSpec(provider=self.provider, model=self.model)

    def to_run_config(
        self,
        *,
        seed: int = 42,
        limit: int | None = None,
        max_cost_usd: float | None = None,
        prompt_profile: str = DEFAULT_PROMPT_PROFILE,
        eval_mode: EvalMode = EvalMode.CONTEXT_GIVEN,
        conversation_protocol: ConversationProtocol = ConversationProtocol.GOLD_HISTORY,
    ) -> RunConfig:
        return RunConfig(
            seed=seed,
            concurrency=self.runtime.concurrency,
            max_retries=self.runtime.retries,
            temperature=self.generation.temperature,
            max_output_tokens=self.generation.max_output_tokens,
            timeout_seconds=self.generation.timeout_seconds,
            cache_mode=CacheMode.READ_WRITE if self.runtime.cache else CacheMode.OFF,
            limit=limit,
            max_cost_usd=max_cost_usd,
            prompt_profile=prompt_profile,
            eval_mode=eval_mode,
            conversation_protocol=conversation_protocol,
        )


def load_model_config(path: str | Path) -> ModelConfigFile:
    """Load and validate a model-config YAML file."""
    file_path = Path(path)
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"could not read model config at {file_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in model config at {file_path}: {exc}") from exc
    return ModelConfigFile.model_validate(raw)
