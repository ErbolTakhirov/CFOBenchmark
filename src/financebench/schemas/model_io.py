"""Model-facing schemas: chat messages, the structured financial-answer envelope, and the
request/response pair the execution engine sends to every provider.

This module is a dependency leaf within the schema package (it only imports ``common`` and
``tooling``) so that ``prediction.py`` and the execution layer can both build on it. Every field
on :class:`ModelRequest` is significant to the response cache — see ``execution/cache.py`` — so
adding a field here that should *not* affect caching must be added to that module's exclusion set.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from financebench.schemas.tooling import ToolSpec

__all__ = [
    "ChatMessage",
    "Citation",
    "FinancialAnswer",
    "ModelRequest",
    "ModelResponse",
    "ModelSpec",
    "Role",
    "TokenUsage",
]


class Role(StrEnum):
    """Speaker role in a chat transcript."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """A single message in a transcript sent to or returned from a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    content: str


class ModelSpec(BaseModel):
    """A parsed ``provider/model`` reference, e.g. ``openai/gpt-4o-mini``.

    The model portion may itself contain slashes (e.g. OpenRouter's
    ``openrouter/meta-llama/llama-3.1-70b-instruct``); only the first slash separates the
    provider from the model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", "model")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("provider and model must be non-empty")
        return stripped

    @classmethod
    def parse(cls, ref: str, *, params: dict[str, Any] | None = None) -> ModelSpec:
        """Parse a ``provider/model`` string into a :class:`ModelSpec`."""
        provider, sep, model = ref.partition("/")
        if not sep or not provider or not model:
            raise ValueError(
                f"Invalid model reference {ref!r}; expected 'provider/model' "
                "(e.g. 'mock/echo-gold', 'openai/gpt-4o-mini')."
            )
        return cls(provider=provider, model=model, params=params or {})

    @property
    def ref(self) -> str:
        """The canonical ``provider/model`` string."""
        return f"{self.provider}/{self.model}"

    @property
    def slug(self) -> str:
        """A filesystem-safe slug for this model reference."""
        return re.sub(r"[^A-Za-z0-9._-]+", "-", self.ref)


class Citation(BaseModel):
    """A pointer a model's answer claims supports it — checked against real evidence at grading
    time by the grounding/hallucination metrics, not trusted at face value."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    document_id: str
    page: int | None = None
    table: str | None = None
    row: str | None = None


class FinancialAnswer(BaseModel):
    """The structured envelope a model returns for a financial question.

    Parsed from the model's raw text by providers via :meth:`from_text`. Extra keys are ignored
    (real models are messy); an answer that cannot be parsed at all is recorded as
    ``parsed=False`` on :class:`ModelResponse` rather than silently defaulted, and unstructured
    text models are still supported through the same best-effort parse.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    answer: str
    numeric_value: float | None = None
    unit: str | None = None
    citations: tuple[Citation, ...] = Field(default_factory=tuple)
    insufficient_information: bool = False
    confidence: float | None = None
    brief_explanation: str | None = None

    @classmethod
    def from_text(cls, text: str) -> FinancialAnswer | None:
        """Best-effort parse of a model's raw text into a :class:`FinancialAnswer`.

        Handles fenced ```json blocks, JSON embedded in surrounding prose, and plain unstructured
        text (falls back to treating the whole trimmed text as the answer). Returns ``None`` only
        when the text is empty.
        """
        candidate = text.strip()
        if not candidate:
            return None
        fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL)
        json_candidate = fence.group(1) if fence else None
        if json_candidate is None:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start != -1 and end > start:
                json_candidate = candidate[start : end + 1]
        if json_candidate is not None:
            try:
                data = json.loads(json_candidate)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                try:
                    return cls.model_validate(data)
                except ValidationError:
                    pass
        # Unstructured fallback: treat the raw text as the answer verbatim so plain-text models
        # are still gradable, just without citations/confidence.
        return cls(answer=candidate)

    def to_json(self) -> str:
        """Serialize to the canonical JSON structured-output providers are asked to emit."""
        return self.model_dump_json()


class TokenUsage(BaseModel):
    """Token accounting as reported by a provider. Missing values stay ``None`` — never invented."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ModelRequest(BaseModel):
    """A single request to a provider's ``generate`` method.

    Every field here is part of the response-cache key (see ``execution/cache.py``) except
    ``request_id`` and ``timeout_s``, which are delivery-mechanism details that must not affect
    whether a cached answer is reused.

    **There is deliberately no channel on this model through which a gold answer can travel.**
    An earlier revision carried a ``simulation_context`` dict holding ``gold_answer`` /
    ``gold_numeric_value`` for the mock provider to read; because a ``ModelRequest`` is serialized
    into ``predictions.jsonl`` *and hashed into the cache key*, that put the answer key one
    forgotten ``if`` away from reaching a real model. The field is gone: the mock now receives its
    oracle by constructor injection (``models/mock.py``), so leakage is prevented by the shape of
    the type rather than by a convention someone has to remember. ``extra="forbid"`` means nobody
    can slip one back in dynamically, and ``tests/security/test_gold_answer_leakage.py`` asserts
    both that no such field exists and that the rendered request is byte-identical when the gold
    answer is replaced with a sentinel.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: ModelSpec
    messages: tuple[ChatMessage, ...]
    temperature: float = 0.0
    max_tokens: int | None = 1024
    response_format: Literal["text", "json_object"] = "json_object"
    tools: tuple[ToolSpec, ...] = Field(default_factory=tuple)
    prompt_version: str
    benchmark: str
    benchmark_version: str
    sample_id: str
    base_url_id: str | None = None
    timeout_s: float = 120.0
    request_id: str | None = None


class ModelResponse(BaseModel):
    """A normalized response from any provider.

    Transport-level failures are raised as exceptions (recorded as failure events by the
    engine); a response whose envelope could not be parsed sets ``financial_answer`` to ``None``
    with ``parsed=False`` so the failure is explicit rather than silently defaulted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str
    content: str
    financial_answer: FinancialAnswer | None = None
    parsed: bool = False
    token_usage: TokenUsage | None = None
    latency_ms: float | None = None
    estimated_cost_usd: float | None = None
    raw: dict[str, Any] | None = None
