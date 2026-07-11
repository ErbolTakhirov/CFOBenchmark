"""The gates, the verdict, and the index.

The property being defended here is the one that motivated the whole design: **a strong average
must not be able to hide a dangerous model.** Every test below is a way of trying to sneak one
past.
"""

from __future__ import annotations

import pytest

from financebench.evaluation.capability_map import CapabilityDimension
from financebench.evaluation.failures import FailureRecord, FailureType
from financebench.evaluation.gates import GATE_THRESHOLDS, Verdict, evaluate_gates, verdict_for
from financebench.evaluation.scoring import compute_scores, reliability_penalty
from financebench.evaluation.stats import bootstrap_ci, paired_bootstrap
from financebench.schemas.common import EvalMode
from financebench.schemas.metric import MetricAggregate


def _failure(failure_type: FailureType, i: int = 0) -> FailureRecord:
    from financebench.evaluation.failures import CATASTROPHIC_FAILURES

    return FailureRecord(
        sample_id=f"finqa:test:{i}",
        benchmark="finqa",
        failure_type=failure_type,
        catastrophic=failure_type in CATASTROPHIC_FAILURES,
    )


def _capabilities(**scores: float) -> dict[CapabilityDimension, MetricAggregate]:
    return {
        CapabilityDimension(name): MetricAggregate(metric_name=name, n=40, mean=value)
        for name, value in scores.items()
    }


# --------------------------------------------------------------------------- gates


def test_a_clean_run_passes_every_gate() -> None:
    gates = evaluate_gates(failures=[], n_scored=100, numeric_accuracy=0.9)
    assert gates.evaluated is True
    assert all(gate.passed for gate in gates.gates)
    assert gates.any_critical_gate_failed is False


def test_a_wrong_scale_rate_above_the_limit_fails_a_critical_gate() -> None:
    """Off by 1000x. In a financial context that is not a near-miss."""
    failures = [_failure(FailureType.WRONG_SCALE, i) for i in range(10)]
    gates = evaluate_gates(failures=failures, n_scored=100, numeric_accuracy=0.9)

    scale_gate = next(g for g in gates.gates if g.gate_name == "wrong_scale_rate_max")
    assert scale_gate.observed == 0.1
    assert scale_gate.passed is False
    assert gates.any_critical_gate_failed is True


def test_inventing_a_number_it_could_not_know_fails_a_critical_gate() -> None:
    failures = [_failure(FailureType.FAILED_REFUSAL, i) for i in range(20)]
    gates = evaluate_gates(failures=failures, n_scored=100, numeric_accuracy=0.9)

    assert next(g for g in gates.gates if g.gate_name == "failed_refusal_rate_max").passed is False
    assert gates.any_critical_gate_failed is True


def test_being_merely_annoying_does_not_fail_a_critical_gate() -> None:
    """An unnecessary refusal is irritating, not dangerous. It gets a looser bound and is not
    critical — conflating the two would make the gates cry wolf."""
    failures = [_failure(FailureType.UNNECESSARY_REFUSAL, i) for i in range(30)]
    gates = evaluate_gates(failures=failures, n_scored=100, numeric_accuracy=0.9)

    unnecessary = next(g for g in gates.gates if g.gate_name == "unnecessary_refusal_rate_max")
    assert unnecessary.passed is False  # it did breach its own limit
    assert gates.any_critical_gate_failed is False  # but it is not a critical gate


def test_a_model_that_cannot_do_arithmetic_fails_the_numeric_accuracy_floor() -> None:
    gates = evaluate_gates(failures=[], n_scored=100, numeric_accuracy=0.30)
    numeric = next(g for g in gates.gates if g.gate_name == "numeric_accuracy_min")
    assert numeric.passed is False
    assert gates.any_critical_gate_failed is True


# --------------------------------------------------------------------------- the verdict


def test_a_critical_gate_failure_caps_the_verdict_even_with_an_excellent_average() -> None:
    """THE test. A model scoring 0.95 that is catastrophically wrong 10% of the time is not a
    0.95 model — it is a model you cannot leave alone with a spreadsheet."""
    failures = [_failure(FailureType.WRONG_SCALE, i) for i in range(10)]
    gates = evaluate_gates(failures=failures, n_scored=100, numeric_accuracy=0.95)

    verdict, reasons = verdict_for(gates=gates, core_score=0.95, n_scored=100)

    assert verdict is Verdict.LIMITED_HIGH_SUPERVISION
    assert verdict is not Verdict.EXCEPTIONAL_BUT_STILL_REQUIRES_CONTROLS
    assert any("critical gate" in reason.lower() for reason in reasons)


def test_the_best_possible_verdict_still_requires_controls() -> None:
    """There is deliberately no 'safe for autonomous financial decisions' label, and no score that
    produces one."""
    gates = evaluate_gates(failures=[], n_scored=100, numeric_accuracy=0.99)
    verdict, _ = verdict_for(gates=gates, core_score=0.97, n_scored=100)

    assert verdict is Verdict.EXCEPTIONAL_BUT_STILL_REQUIRES_CONTROLS
    assert "REQUIRES_CONTROLS" in verdict.value


def test_too_few_samples_yields_insufficient_coverage_not_a_flattering_score() -> None:
    gates = evaluate_gates(failures=[], n_scored=5, numeric_accuracy=1.0)
    verdict, reasons = verdict_for(gates=gates, core_score=1.0, n_scored=5)

    assert verdict is Verdict.INSUFFICIENT_COVERAGE
    assert any("not enough evidence" in reason for reason in reasons)


def test_a_mock_run_can_never_receive_a_readiness_verdict() -> None:
    gates = evaluate_gates(failures=[], n_scored=100, numeric_accuracy=1.0)
    verdict, reasons = verdict_for(gates=gates, core_score=1.0, n_scored=100, is_mock=True)

    assert verdict is Verdict.NOT_EVALUATED
    assert any("no model was evaluated" in reason.lower() for reason in reasons)


@pytest.mark.parametrize(
    ("core", "expected"),
    [
        (0.20, Verdict.NOT_FINANCE_READY),
        (0.45, Verdict.LIMITED_HIGH_SUPERVISION),
        (0.65, Verdict.USABLE_WITH_HUMAN_REVIEW),
        (0.80, Verdict.STRONG_FOR_BOUNDED_FINANCIAL_TASKS),
        (0.95, Verdict.EXCEPTIONAL_BUT_STILL_REQUIRES_CONTROLS),
    ],
)
def test_the_verdict_bands(core: float, expected: Verdict) -> None:
    gates = evaluate_gates(failures=[], n_scored=100, numeric_accuracy=max(core, 0.6))
    verdict, _ = verdict_for(gates=gates, core_score=core, n_scored=100)
    assert verdict is expected


# --------------------------------------------------------------------------- the index


def test_the_geometric_mean_refuses_to_average_away_a_catastrophic_weakness() -> None:
    """The reason the FCI is geometric and not arithmetic.

    A model with 0.9 grounding and 0.1 numerical accuracy has an arithmetic mean of 0.5 — which
    reads as "mediocre but usable". It cannot do arithmetic. The geometric mean says so.
    """
    scores = compute_scores(
        eval_mode=EvalMode.CONTEXT_GIVEN,
        capabilities=_capabilities(
            numerical_accuracy=0.1, document_grounding=0.9, table_text_reasoning=0.9
        ),
        failures=[],
        n_scored=100,
        any_critical_gate_failed=False,
        is_mock=False,
    )
    assert scores.fci is not None
    assert scores.fci < 0.5, "a geometric mean must not let a strength buy off a fatal weakness"


def test_the_index_is_withheld_when_a_critical_gate_failed() -> None:
    scores = compute_scores(
        eval_mode=EvalMode.CONTEXT_GIVEN,
        capabilities=_capabilities(
            numerical_accuracy=0.9, document_grounding=0.9, table_text_reasoning=0.9
        ),
        failures=[],
        n_scored=100,
        any_critical_gate_failed=True,
        is_mock=False,
    )
    assert scores.fci is None
    assert scores.fci_withheld_because is not None
    assert "critical gate" in scores.fci_withheld_because


def test_the_index_is_withheld_for_a_mock_run() -> None:
    scores = compute_scores(
        eval_mode=EvalMode.CONTEXT_GIVEN,
        capabilities=_capabilities(
            numerical_accuracy=1.0, document_grounding=1.0, table_text_reasoning=1.0
        ),
        failures=[],
        n_scored=100,
        any_critical_gate_failed=False,
        is_mock=True,
    )
    assert scores.fci is None
    assert "mock" in (scores.fci_withheld_because or "")


def test_a_context_given_run_reports_no_rag_or_agent_score() -> None:
    """It said nothing about a retriever. It must not imply that it did."""
    scores = compute_scores(
        eval_mode=EvalMode.CONTEXT_GIVEN,
        capabilities=_capabilities(numerical_accuracy=0.8),
        failures=[],
        n_scored=100,
        any_critical_gate_failed=False,
        is_mock=False,
    )
    assert scores.core_score is not None
    assert scores.rag_score is None
    assert scores.agent_score is None


def test_the_reliability_penalty_punishes_dangerous_failure_more_than_frequent_failure() -> None:
    merely_wrong = [_failure(FailureType.WRONG_NUMBER, i) for i in range(50)]
    catastrophic = [_failure(FailureType.WRONG_SCALE, i) for i in range(10)]

    assert reliability_penalty(merely_wrong, 100) == 1.0, (
        "being wrong is not the same as being dangerous"
    )
    assert reliability_penalty(catastrophic, 100) < 1.0


# --------------------------------------------------------------------------- statistics


def test_a_confidence_interval_brackets_the_mean_and_flags_a_small_sample() -> None:
    result = bootstrap_ci([1.0] * 5 + [0.0] * 5)
    assert result is not None
    assert result.ci_low <= result.mean <= result.ci_high
    assert result.underpowered is True, "10 samples cannot support a claim"


def test_the_bootstrap_is_deterministic() -> None:
    """An interval that wobbles between invocations is not an interval."""
    values = [1.0, 0.0, 1.0, 1.0, 0.0] * 10
    assert bootstrap_ci(values) == bootstrap_ci(values)


def test_two_models_that_differ_by_noise_are_not_declared_different() -> None:
    a = {f"s{i}": float(i % 2) for i in range(40)}  # 50%
    b = {f"s{i}": float((i + 1) % 2) for i in range(40)}  # 50%, different questions

    comparison = paired_bootstrap(a, b)
    assert comparison is not None
    assert comparison.significant is False
    assert "No significant difference" in comparison.verdict("A", "B")


def test_a_real_difference_is_detected_when_pairing_reveals_it() -> None:
    """B gets everything A gets, plus more. Pairing sees this; an unpaired test would struggle."""
    a = {f"s{i}": (1.0 if i < 10 else 0.0) for i in range(50)}
    b = {f"s{i}": (1.0 if i < 35 else 0.0) for i in range(50)}

    comparison = paired_bootstrap(a, b)
    assert comparison is not None
    assert comparison.significant is True
    assert comparison.mean_difference < 0  # b is better
    assert "beats" in comparison.verdict("A", "B")


def test_a_comparison_on_too_few_pairs_makes_no_claim() -> None:
    a = {f"s{i}": 1.0 for i in range(5)}
    b = {f"s{i}": 0.0 for i in range(5)}

    comparison = paired_bootstrap(a, b)
    assert comparison is not None
    assert comparison.underpowered is True
    assert comparison.significant is False
    assert "Too few paired samples" in comparison.verdict("A", "B")


def test_gate_thresholds_are_stated_so_they_can_be_argued_with() -> None:
    """They are judgements, not measurements. Burying them in a scoring function would make them
    unfalsifiable."""
    assert GATE_THRESHOLDS["catastrophic_numeric_error_rate_max"] == 0.05
    assert GATE_THRESHOLDS["numeric_accuracy_min"] == 0.50


def test_a_model_that_scores_zero_is_not_reported_as_never_run() -> None:
    """A legitimate score of 0.0 is FALSY in Python.

    `core_score or rag_score or agent_score` therefore turns a real 0.0 into None, and the verdict
    becomes NOT_EVALUATED — making the worst possible model indistinguishable from one that was
    never run. Seen for real: qwen2.5:3b scored exactly 0.000 on FinanceReasoning-hard. That is a
    true and important result, and the report erased it.
    """
    gates = evaluate_gates(failures=[], n_scored=40, numeric_accuracy=0.0)
    verdict, _ = verdict_for(gates=gates, core_score=0.0, n_scored=40)

    assert verdict is not Verdict.NOT_EVALUATED
    assert verdict is Verdict.NOT_FINANCE_READY, "scoring zero is a finding, not an absence"
