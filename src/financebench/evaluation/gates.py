"""Critical gates, and the finance-readiness verdict.

A mean score is a bad summary of a financial model, because it treats all errors as the same size.
They are not. Being off by 2 % is a rounding disagreement; being off by 1000x because you confused
thousands with millions is a disaster, and it *looks* exactly the same in a mean.

So gates are evaluated **independently of the average**, and a failed gate can override an
otherwise-strong score. A model that answers 85 % of questions correctly and confidently invents a
number on the other 15 % is not an 85 % model — it is a model you cannot leave alone with a
spreadsheet.

The verdict tops out at ``EXCEPTIONAL_BUT_STILL_REQUIRES_CONTROLS``. There is deliberately no
"safe for autonomous financial decisions" label, and no threshold that produces one. A benchmark
measures a benchmark.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from financebench.evaluation.failures import (
    CATASTROPHIC_FAILURES,
    FailureRecord,
    FailureType,
)
from financebench.schemas.gates import GateResult, GatesReport

__all__ = [
    "GATE_THRESHOLDS",
    "Verdict",
    "evaluate_gates",
    "verdict_for",
]


class Verdict(StrEnum):
    """The finance-readiness label. Ordered from worst to best."""

    NOT_EVALUATED = "NOT_EVALUATED"
    INSUFFICIENT_COVERAGE = "INSUFFICIENT_COVERAGE"
    NOT_FINANCE_READY = "NOT_FINANCE_READY"
    LIMITED_HIGH_SUPERVISION = "LIMITED_HIGH_SUPERVISION"
    USABLE_WITH_HUMAN_REVIEW = "USABLE_WITH_HUMAN_REVIEW"
    STRONG_FOR_BOUNDED_FINANCIAL_TASKS = "STRONG_FOR_BOUNDED_FINANCIAL_TASKS"
    EXCEPTIONAL_BUT_STILL_REQUIRES_CONTROLS = "EXCEPTIONAL_BUT_STILL_REQUIRES_CONTROLS"


#: Every gate is a *maximum permitted rate*, except ``numeric_accuracy`` which is a minimum.
#: These are judgements, not measurements — they are stated here so they can be argued with, rather
#: than buried in a scoring function.
GATE_THRESHOLDS: dict[str, float] = {
    # A model that cannot do arithmetic reliably cannot do finance at all.
    "numeric_accuracy_min": 0.50,
    # The one that matters most: confidently, catastrophically wrong (scale, sign, %-vs-pp, bps).
    "catastrophic_numeric_error_rate_max": 0.05,
    # Answering a question the context cannot support — inventing a number.
    "failed_refusal_rate_max": 0.10,
    # Refusing a question it could have answered. Annoying, not dangerous — a looser bound.
    "unnecessary_refusal_rate_max": 0.25,
    "wrong_scale_rate_max": 0.03,
    "wrong_currency_rate_max": 0.02,
    "wrong_period_rate_max": 0.05,
    "unsupported_claim_rate_max": 0.10,
    # Malformed output is a real cost — but it is a formatting failure, not a reasoning one, so it
    # is bounded separately rather than being conflated with getting the number wrong.
    "invalid_output_rate_max": 0.10,
}

#: Gates whose failure alone caps the verdict, no matter how good the average is.
_CRITICAL = frozenset(
    {
        "numeric_accuracy_min",
        "catastrophic_numeric_error_rate_max",
        "failed_refusal_rate_max",
        "wrong_scale_rate_max",
        "wrong_currency_rate_max",
    }
)


def _rate(failures: Sequence[FailureRecord], types: frozenset[FailureType], total: int) -> float:
    if total == 0:
        return 0.0
    return sum(1 for f in failures if f.failure_type in types) / total


def evaluate_gates(
    *,
    failures: Sequence[FailureRecord],
    n_scored: int,
    numeric_accuracy: float | None,
) -> GatesReport:
    """Evaluate every gate against a run's failure records.

    ``n_scored`` is the number of samples that were actually *graded* — infrastructure failures
    (timeouts, provider errors) are excluded, because scoring our own network problems as the
    model's financial failings would be a lie about the model.
    """
    if n_scored == 0:
        return GatesReport(evaluated=False)

    def gate(name: str, observed: float, *, minimum: bool = False) -> GateResult:
        threshold = GATE_THRESHOLDS[name]
        passed = observed >= threshold if minimum else observed <= threshold
        return GateResult(
            gate_name=name, threshold=threshold, observed=round(observed, 4), passed=passed
        )

    results = [
        gate("numeric_accuracy_min", numeric_accuracy or 0.0, minimum=True),
        gate(
            "catastrophic_numeric_error_rate_max",
            _rate(failures, CATASTROPHIC_FAILURES, n_scored),
        ),
        gate(
            "failed_refusal_rate_max",
            _rate(failures, frozenset({FailureType.FAILED_REFUSAL}), n_scored),
        ),
        gate(
            "unnecessary_refusal_rate_max",
            _rate(failures, frozenset({FailureType.UNNECESSARY_REFUSAL}), n_scored),
        ),
        gate(
            "wrong_scale_rate_max", _rate(failures, frozenset({FailureType.WRONG_SCALE}), n_scored)
        ),
        gate(
            "wrong_currency_rate_max",
            _rate(failures, frozenset({FailureType.WRONG_CURRENCY}), n_scored),
        ),
        gate(
            "wrong_period_rate_max",
            _rate(failures, frozenset({FailureType.WRONG_PERIOD}), n_scored),
        ),
        gate(
            "unsupported_claim_rate_max",
            _rate(
                failures,
                frozenset(
                    {
                        FailureType.UNSUPPORTED_NUMERIC_CLAIM,
                        FailureType.UNSUPPORTED_NARRATIVE_CLAIM,
                    }
                ),
                n_scored,
            ),
        ),
        gate(
            "invalid_output_rate_max",
            _rate(failures, frozenset({FailureType.INVALID_STRUCTURED_RESPONSE}), n_scored),
        ),
    ]

    critical_failed = any(r.gate_name in _CRITICAL and r.passed is False for r in results)
    return GatesReport(
        evaluated=True, gates=tuple(results), any_critical_gate_failed=critical_failed
    )


def verdict_for(
    *,
    gates: GatesReport,
    core_score: float | None,
    n_scored: int,
    min_samples: int = 30,
    is_mock: bool = False,
) -> tuple[Verdict, list[str]]:
    """Derive the finance-readiness verdict. Returns ``(verdict, reasons)``.

    The reasons are as important as the label — a verdict with no stated basis is an opinion.
    """
    reasons: list[str] = []

    if is_mock:
        return Verdict.NOT_EVALUATED, [
            "This run used the mock provider, which is handed the gold answers. No model was "
            "evaluated, so no readiness claim of any kind can be made."
        ]

    if not gates.evaluated or core_score is None or n_scored == 0:
        return Verdict.NOT_EVALUATED, ["No samples were successfully scored."]

    if n_scored < min_samples:
        return Verdict.INSUFFICIENT_COVERAGE, [
            f"Only {n_scored} samples were scored (minimum {min_samples} for any claim). "
            "The score below is real, but it is not enough evidence to characterise a model."
        ]

    failed = [g for g in gates.gates if g.passed is False]
    critical_failed = [g for g in failed if g.gate_name in _CRITICAL]

    for gate_result in failed:
        comparator = "below" if gate_result.gate_name.endswith("_min") else "above"
        reasons.append(
            f"FAILED GATE {gate_result.gate_name}: observed {gate_result.observed}, "
            f"{comparator} the limit of {gate_result.threshold}."
        )

    # A critical gate failure caps the verdict regardless of the average. This is the whole point:
    # a strong mean must not be able to hide a model that is confidently, catastrophically wrong.
    if critical_failed:
        reasons.insert(
            0,
            "A critical gate failed. However good the average is, this model makes the kind of "
            "error that is not a near-miss in a financial context.",
        )
        return (
            Verdict.NOT_FINANCE_READY if core_score < 0.60 else Verdict.LIMITED_HIGH_SUPERVISION
        ), reasons

    if core_score < 0.35:
        reasons.append(f"Core score {core_score:.2f} — wrong more often than right.")
        return Verdict.NOT_FINANCE_READY, reasons
    if core_score < 0.55:
        reasons.append(f"Core score {core_score:.2f} — every answer needs checking.")
        return Verdict.LIMITED_HIGH_SUPERVISION, reasons
    if core_score < 0.75:
        reasons.append(f"Core score {core_score:.2f} — useful, but a human must review the output.")
        return Verdict.USABLE_WITH_HUMAN_REVIEW, reasons
    if core_score < 0.90:
        reasons.append(
            f"Core score {core_score:.2f} — strong on the task types measured here. That is not "
            "the same as strong on your task types."
        )
        return Verdict.STRONG_FOR_BOUNDED_FINANCIAL_TASKS, reasons

    reasons.append(
        f"Core score {core_score:.2f} with no failed gates. Still requires controls: a benchmark "
        "measures a benchmark, and this one cannot tell you what the model does on data it has "
        "never seen, in a workflow it was not tested in."
    )
    return Verdict.EXCEPTIONAL_BUT_STILL_REQUIRES_CONTROLS, reasons
