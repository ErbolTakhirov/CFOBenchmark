"""Deterministic failure attribution.

"The model got 43 % " tells you almost nothing. *How* it fails is the whole question — a model that
is off by a rounding error is a different proposition from one that is off by a factor of a
thousand, and a model that confidently invents a number is a different proposition again from one
that says it doesn't know.

So every failed sample is classified, and the classification is **derived from the numbers**, not
guessed by a judge. Where a failure genuinely cannot be attributed it becomes ``WRONG_NUMBER``
rather than being forced into a bucket that flatters the taxonomy.

The dangerous classes — the ones that feed the critical gates — are the ones where the model is
confidently, catastrophically wrong: a wrong scale (out by 1000x), a %-vs-percentage-point mix-up,
a sign error. In a financial context those are not near-misses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from financebench.evaluation.numeric import parse_numeric_answer
from financebench.schemas.metric import MetricResult
from financebench.schemas.prediction import Prediction
from financebench.schemas.sample import CanonicalSample

__all__ = [
    "CATASTROPHIC_FAILURES",
    "FailureRecord",
    "FailureType",
    "attribute_failure",
    "failure_distribution",
]


class FailureType(StrEnum):
    """How a prediction went wrong."""

    # -- numeric
    WRONG_NUMBER = "wrong_number"
    WRONG_SIGN = "wrong_sign"
    WRONG_MAGNITUDE = "wrong_magnitude"  # out by a power of ten
    WRONG_SCALE = "wrong_scale"  # out by exactly 1e3 / 1e6 / 1e9
    PERCENT_VS_PERCENTAGE_POINT = "percent_vs_percentage_point"
    BASIS_POINT_CONFUSION = "basis_point_confusion"
    WRONG_UNIT = "wrong_unit"
    WRONG_CURRENCY = "wrong_currency"
    WRONG_PERIOD = "wrong_period"
    NO_EXTRACTABLE_NUMBER = "no_extractable_number"

    # -- reasoning
    WRONG_FORMULA = "wrong_formula"

    # -- grounding
    UNSUPPORTED_NUMERIC_CLAIM = "unsupported_numeric_claim"
    UNSUPPORTED_NARRATIVE_CLAIM = "unsupported_narrative_claim"
    WRONG_EVIDENCE = "wrong_evidence"
    WRONG_DOCUMENT = "wrong_document"
    WRONG_PAGE = "wrong_page"
    RETRIEVAL_MISS = "retrieval_miss"
    GENERATION_ERROR_AFTER_RETRIEVAL = "generation_error_after_retrieval"

    # -- calibration
    UNNECESSARY_REFUSAL = "unnecessary_refusal"
    FAILED_REFUSAL = "failed_refusal"

    # -- conversation
    CONVERSATION_CONTEXT_LOSS = "conversation_context_loss"

    # -- protocol
    INVALID_STRUCTURED_RESPONSE = "invalid_structured_response"

    # -- tools
    TOOL_SELECTION_ERROR = "tool_selection_error"
    TOOL_ARGUMENT_ERROR = "tool_argument_error"
    TOOL_RESULT_IGNORED = "tool_result_ignored"

    # -- infrastructure (not the model's fault; must not be scored as if it were)
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_ERROR = "provider_error"


#: Failures where the model was confidently, dangerously wrong — not merely inaccurate. A model
#: with a good mean score and a meaningful rate of *these* is not a safe model, and the critical
#: gates exist to stop a strong average from hiding them.
CATASTROPHIC_FAILURES: frozenset[FailureType] = frozenset(
    {
        FailureType.WRONG_SCALE,
        FailureType.WRONG_MAGNITUDE,
        FailureType.WRONG_SIGN,
        FailureType.PERCENT_VS_PERCENTAGE_POINT,
        FailureType.BASIS_POINT_CONFUSION,
        FailureType.WRONG_CURRENCY,
        FailureType.FAILED_REFUSAL,
    }
)

#: Infrastructure failures. These are OUR problem, not the model's, and are excluded from
#: capability scores — scoring a network timeout as a financial-reasoning failure would be a lie
#: about the model.
INFRASTRUCTURE_FAILURES: frozenset[FailureType] = frozenset(
    {FailureType.PROVIDER_TIMEOUT, FailureType.PROVIDER_ERROR}
)


class FailureRecord(BaseModel):
    """One failed sample, classified — written to ``failures.jsonl``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str
    benchmark: str
    failure_type: FailureType
    catastrophic: bool = False
    question: str = ""
    gold: str = ""
    predicted: str = ""
    detail: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


_SCALE_FACTORS: dict[float, str] = {1e3: "thousand", 1e6: "million", 1e9: "billion"}
_TOLERANCE = 0.01  # 1 % — loose enough to see through rounding, tight enough to be a real match


def _close(a: float, b: float, tolerance: float = _TOLERANCE) -> bool:
    if b == 0:
        return abs(a) < 1e-9
    return abs(a - b) / abs(b) <= tolerance


def _predicted_number(prediction: Prediction) -> float | None:
    response = prediction.response
    if response is None or response.financial_answer is None:
        return None
    answer = response.financial_answer
    if answer.numeric_value is not None:
        return answer.numeric_value
    parsed = parse_numeric_answer(answer.answer)
    return parsed.resolved_value if parsed is not None else None


def _classify_numeric(predicted: float, gold: float) -> FailureType:
    """Work out *how* a number is wrong, in order of how alarming the explanation is."""
    if gold == 0:
        return FailureType.WRONG_NUMBER

    # Right digits, wrong sign. In finance the difference between a profit and a loss.
    if _close(abs(predicted), abs(gold)) and (predicted < 0) != (gold < 0):
        return FailureType.WRONG_SIGN

    ratio = predicted / gold if gold != 0 else math.inf

    # Out by exactly a reporting scale: the answer was right, the magnitude was not.
    for factor in _SCALE_FACTORS:
        if _close(ratio, factor) or _close(ratio, 1 / factor):
            return FailureType.WRONG_SCALE

    # 0.2342 vs 23.42 — a rate reported as a fraction, or vice versa.
    if _close(ratio, 100.0) or _close(ratio, 0.01):
        return FailureType.PERCENT_VS_PERCENTAGE_POINT

    # 250 bps vs 2.5 %.
    if _close(ratio, 10_000.0) or _close(ratio, 1 / 10_000.0):
        return FailureType.BASIS_POINT_CONFUSION

    # Any other clean power of ten.
    if ratio > 0:
        exponent = math.log10(ratio)
        if abs(exponent - round(exponent)) < 0.01 and round(exponent) != 0:
            return FailureType.WRONG_MAGNITUDE

    return FailureType.WRONG_NUMBER


def attribute_failure(
    sample: CanonicalSample, prediction: Prediction, result: MetricResult | None = None
) -> FailureRecord | None:
    """Classify a failed prediction. Returns ``None`` when nothing failed."""
    response = prediction.response

    # -- infrastructure: not the model's failing, and must not be scored as one.
    if response is None:
        failure_type = (
            FailureType.PROVIDER_TIMEOUT
            if prediction.error_type and "timeout" in prediction.error_type.lower()
            else FailureType.PROVIDER_ERROR
        )
        return FailureRecord(
            sample_id=sample.sample_id,
            benchmark=sample.benchmark,
            failure_type=failure_type,
            question=sample.question[:300],
            gold=sample.gold.answer[:200],
            detail=(prediction.error or "")[:300],
        )

    # passed=None means NOT APPLICABLE, not "failed". A free-text FinanceBench answer that no
    # deterministic metric can check is not a model failure — recording it as one would invent a
    # 0.0 out of our own inability to grade it.
    if result is not None and result.passed is not False:
        return None

    answer = response.financial_answer
    if answer is None:
        return FailureRecord(
            sample_id=sample.sample_id,
            benchmark=sample.benchmark,
            failure_type=FailureType.INVALID_STRUCTURED_RESPONSE,
            question=sample.question[:300],
            gold=sample.gold.answer[:200],
            predicted=response.content[:200],
            detail="the response could not be parsed into the answer envelope at all",
        )

    def record(failure_type: FailureType, detail: str = "") -> FailureRecord:
        return FailureRecord(
            sample_id=sample.sample_id,
            benchmark=sample.benchmark,
            failure_type=failure_type,
            catastrophic=failure_type in CATASTROPHIC_FAILURES,
            question=sample.question[:300],
            gold=sample.gold.answer[:200],
            predicted=(answer.answer or "")[:200],
            detail=detail,
            metadata={"task_family": sample.task_family},
        )

    # -- calibration: did it refuse, and should it have?
    should_refuse = sample.evaluation.should_refuse
    refused = answer.insufficient_information
    if refused and not should_refuse:
        return record(
            FailureType.UNNECESSARY_REFUSAL,
            "declined to answer a question that the context does support",
        )
    if should_refuse and not refused:
        return record(
            FailureType.FAILED_REFUSAL,
            "answered a question the context cannot support — the most dangerous failure mode",
        )
    if refused and should_refuse:
        return None  # a correct refusal is not a failure

    # -- numeric
    gold_value = sample.gold.numeric_value
    predicted_value = _predicted_number(prediction)

    if gold_value is not None and predicted_value is None:
        return record(
            FailureType.NO_EXTRACTABLE_NUMBER,
            "a numeric answer was expected but none could be extracted from the response",
        )
    if gold_value is not None and predicted_value is not None:
        failure_type = _classify_numeric(predicted_value, gold_value)
        return record(failure_type, f"predicted {predicted_value!r}, gold {gold_value!r}")

    # -- textual
    return record(FailureType.WRONG_NUMBER, "answer does not match gold")


def failure_distribution(records: Sequence[FailureRecord]) -> dict[str, int]:
    """Counts per failure type, sorted most-common-first."""
    counts: dict[str, int] = {}
    for record in records:
        counts[record.failure_type.value] = counts.get(record.failure_type.value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
