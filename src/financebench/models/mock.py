"""Deterministic mock provider — a *simulator*, not a model under test.

The mock answers from an **oracle injected at construction time** (``MockProvider(oracle=...)``),
built by the execution layer from the samples about to be run. The oracle never travels inside a
:class:`~financebench.schemas.model_io.ModelRequest` — that type has no field it could travel in
(see its docstring). A mock built without an oracle simply cannot answer, which is the safe
failure: the only way for the answer key to reach this provider is for a caller to hand it over
explicitly, in code, on the mock-only path.

The profiles exist to *differentiate* behaviour, so scoring is demonstrably meaningful rather than
trivially 100 % or 0 %:

- ``echo-gold`` returns the exact gold answer verbatim (the happy path).
- ``formatting-noise`` returns the right number wrapped in messy prose (commas, currency symbols,
  percent signs, parenthesized negatives) — exercises the numeric parser, not the mock.
- ``always-wrong`` returns a deterministic, obviously-wrong numeric answer.
- ``refuse`` always declines to answer (exercises refusal / ``should_refuse`` grading).
- ``error`` / ``timeout`` deterministically raise a non-retryable / retryable
  :class:`~financebench.utils.errors.ProviderError` (exercises backoff and ``errors.jsonl``).

**Because the mock can see the answer key, mock scores validate the pipeline, not model quality.**
Runs using it are stamped ``run_type="mock_test"``, are barred from the leaderboard and from the
Finance Capability Index, and require an explicit ``--allow-mock`` on the CLI.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from financebench.models.base import ModelProvider, ProviderCapabilities, register_provider
from financebench.schemas.model_io import FinancialAnswer, ModelRequest, ModelResponse, TokenUsage
from financebench.schemas.sample import CanonicalSample
from financebench.utils.errors import ProviderResponseError, ProviderTimeoutError

__all__ = ["GoldOracleEntry", "MockProvider", "build_mock_oracle"]


def _tok(text: str) -> int:
    """A crude but deterministic token estimate (~4 chars/token)."""
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class GoldOracleEntry:
    """One sample's answer key, as handed to the mock simulator — and to nothing else."""

    gold_answer: str = ""
    gold_numeric_value: float | None = None
    unit: str | None = None
    scale: str | None = None

    @classmethod
    def from_sample(cls, sample: CanonicalSample) -> GoldOracleEntry:
        return cls(
            gold_answer=sample.gold.answer,
            gold_numeric_value=sample.gold.numeric_value,
            unit=sample.gold.unit,
            scale=sample.gold.scale.value if sample.gold.scale is not None else None,
        )

    @property
    def stated_unit(self) -> str | None:
        """What a *perfect* model would write in the answer's ``unit`` field.

        The canonical schema keeps magnitude (``scale``: thousand/million/billion) separate from
        unit (``percent``, ``usd``, ...). A model has no such separation — it writes one word. If
        the mock only ever echoed ``unit``, a million-scale answer would come back with no
        magnitude at all, and would then score *zero* against a benchmark like TAT-QA that folds
        the magnitude into the compared value. That would look like a metric bug and is really a
        simulator that cannot express a correct answer.
        """
        return self.unit or self.scale


def build_mock_oracle(samples: Sequence[CanonicalSample]) -> dict[str, GoldOracleEntry]:
    """Build the answer oracle for a mock run, keyed by ``sample_id``.

    Call this **only** on the mock path. It is the single place in the codebase where gold answers
    are handed to something that will produce a "prediction".
    """
    return {sample.sample_id: GoldOracleEntry.from_sample(sample) for sample in samples}


def _messy_number(value: float | None, unit: str | None) -> str:
    """Render a number the way a real model's prose often does: commas, currency signs, percent
    signs, and parenthesized negatives — deliberately hostile to a naive ``float()`` call so the
    numeric parser has something real to prove itself against."""
    if value is None:
        return "an unspecified amount"
    if unit == "percent":
        return f"{value:.2f}%"
    magnitude = abs(value)
    formatted = f"{magnitude:,.2f}"
    if unit in {"usd", "usd_millions", "usd_thousands", "usd_billions"}:
        formatted = f"${formatted}"
    return f"({formatted})" if value < 0 else formatted


def _echo_gold(entry: GoldOracleEntry) -> FinancialAnswer:
    return FinancialAnswer(
        answer=entry.gold_answer,
        numeric_value=entry.gold_numeric_value,
        unit=entry.stated_unit,
        brief_explanation="Directly taken from the referenced data.",
    )


def _formatting_noise(entry: GoldOracleEntry) -> FinancialAnswer:
    noisy = _messy_number(entry.gold_numeric_value, entry.unit)
    return FinancialAnswer(
        answer=f"Based on the filing, the figure comes out to approximately {noisy}, "
        "per the table referenced above.",
        numeric_value=entry.gold_numeric_value,
        unit=entry.stated_unit,
        brief_explanation="Derived from the referenced table.",
    )


def _always_wrong(entry: GoldOracleEntry) -> FinancialAnswer:
    """Deterministically, *unconditionally* wrong.

    This used to be ``gold + 999``, which is not wrong enough. Benchmarks grade with **relative**
    tolerances — FinanceReasoning's is 0.2 % of the gold — so for a gold of 1,500,000 the tolerance
    is ±3,000 and ``gold + 999`` lands comfortably *inside* it. A profile called ``always-wrong``
    was therefore sometimes right, which quietly undermines every test that asserts a wrong answer
    scores zero.

    Doubling and offsetting is wrong by ~100 % of the gold, which no sane tolerance admits, and
    still lands somewhere wrong when the gold is 0.
    """
    gold = entry.gold_numeric_value or 0.0
    wrong_value = gold * 2.0 + 1000.0
    return FinancialAnswer(
        answer=str(wrong_value), numeric_value=wrong_value, unit=entry.stated_unit
    )


def _refuse(_: GoldOracleEntry) -> FinancialAnswer:
    return FinancialAnswer(
        answer="I don't have enough information in the provided context to answer this.",
        insufficient_information=True,
    )


_ANSWER_PROFILES: dict[str, Callable[[GoldOracleEntry], FinancialAnswer]] = {
    "echo-gold": _echo_gold,
    "formatting-noise": _formatting_noise,
    "always-wrong": _always_wrong,
    "refuse": _refuse,
}

# Fixed simulated latencies keep artifacts byte-stable regardless of the host clock.
_LATENCY_MS: dict[str, float] = {
    "echo-gold": 5.0,
    "formatting-noise": 6.0,
    "always-wrong": 4.0,
    "refuse": 3.0,
}


@register_provider("mock")
class MockProvider(ModelProvider):
    """A deterministic, offline financial-answer simulator with selectable profiles.

    ``oracle`` maps ``sample_id`` to that sample's answer key. Constructed without one (e.g. via
    the generic :func:`~financebench.models.base.create_provider`), the mock has nothing to echo
    and answers emptily — deliberately, so that the only way to get a "correct" mock answer is to
    hand it the oracle on purpose.
    """

    provider = "mock"

    PROFILES: ClassVar[tuple[str, ...]] = (*_ANSWER_PROFILES, "error", "timeout")

    def __init__(self, oracle: Mapping[str, GoldOracleEntry] | None = None) -> None:
        self._oracle: dict[str, GoldOracleEntry] = dict(oracle or {})

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> MockProvider:
        return cls()

    def capabilities(self, model: str) -> ProviderCapabilities:
        return ProviderCapabilities(
            text=True,
            json_mode=True,
            streaming=True,
            max_context_tokens=32_768,
            reports_usage=True,
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        profile = request.model.model
        if profile not in self.PROFILES:
            raise ProviderResponseError(
                f"unknown mock profile {profile!r}; available: {list(self.PROFILES)}",
                provider="mock",
                retryable=False,
            )
        if profile == "timeout":
            raise ProviderTimeoutError(
                "mock/timeout deliberately times out", provider="mock", retryable=True
            )
        if profile == "error":
            raise ProviderResponseError(
                "mock/error deliberately fails", provider="mock", retryable=False
            )

        entry = self._oracle.get(request.sample_id, GoldOracleEntry())
        answer = _ANSWER_PROFILES[profile](entry)
        content = answer.to_json()
        prompt_tokens = sum(_tok(m.content) for m in request.messages)
        completion_tokens = _tok(content)
        return ModelResponse(
            provider="mock",
            model=profile,
            content=content,
            financial_answer=answer,
            parsed=True,
            token_usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_ms=_LATENCY_MS.get(profile, 5.0),
            estimated_cost_usd=0.0,
            raw={"profile": profile},
        )
