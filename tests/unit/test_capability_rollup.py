"""The capability rollup — where "not applicable" was quietly becoming "failed".

Two bugs lived here, and both of them invented failures the model never committed. They were found by
reading a real report and noticing it contradicted itself.

**1. `None` was being scored as `0.0`.**

A metric returns ``passed=None`` to say *"this question cannot be graded by me"*. FinanceBench's 61
analytical questions have no deterministically checkable answer; SMB-CFO's accuracy metric cannot
grade a question the books cannot answer. The rollup turned every one of those into a **zero** and
fed them into the capability scores, the FCI, and the verdict.

On the real 150-question FinanceBench run, document grounding was reported as **0.151**. The true
figure, over the 89 questions that can actually be graded, is **0.254**. The model was understated by
68 % relative — and in the one direction nobody thinks to check, because a low score looks like a
finding rather than a bug.

**2. A dimension was scored by a metric that cannot measure it.**

The real SMB-CFO report contained both of these numbers, in the same file:

    smb_cfo_refusal_correctness   1.000     <- it declined every unanswerable question
    calibration_and_refusal       0.000     <- "it cannot refuse"

Every dimension was fed the benchmark's *preferred* metric, which for SMB-CFO is accuracy. Accuracy
on an unanswerable question is not applicable — there is no number to get right — so the refusal
dimension was scored on a metric that could not apply to the samples in it, and (via bug 1) every one
of those became a zero. A model that refused perfectly was reported as incapable of refusing.

That is the second time in this project that the most important safety metric came out inverted, by a
completely different mechanism. Hence this file.
"""

from __future__ import annotations

import pytest
from tests.factories import make_sample

from financebench.evaluation.capability_map import CapabilityDimension, rollup_capabilities
from financebench.schemas.metric import MetricResult
from financebench.schemas.sample import CanonicalSample


def _result(sample_id: str, metric: str, passed: bool | None) -> MetricResult:
    return MetricResult(
        sample_id=sample_id,
        metric_name=metric,
        value=passed,
        passed=passed,
    )


def _sample(sample_id: str, tags: tuple[str, ...]) -> CanonicalSample:
    return make_sample(sample_id=sample_id, capability_tags=tags)


# --------------------------------------------------------------------------- None is not zero


def test_a_not_applicable_result_is_excluded_not_scored_as_zero() -> None:
    """The bug, in its simplest form.

    Two questions the model got right, and two nobody can grade. The score is 1.0 — not 0.5.
    """
    samples = [_sample(f"smoke:dev:{i}", ("calculation",)) for i in range(4)]
    results = [
        _result("smoke:dev:0", "exact_match", True),
        _result("smoke:dev:1", "exact_match", True),
        _result("smoke:dev:2", "exact_match", None),  # not gradable
        _result("smoke:dev:3", "exact_match", None),  # not gradable
    ]

    rollup = rollup_capabilities(samples, results)
    numeric = rollup[CapabilityDimension.NUMERICAL_ACCURACY]

    assert numeric.mean == 1.0, "the two ungradeable questions must not drag the score to 0.5"
    assert numeric.n == 2, "n must report the evidence the score actually rests on"


def test_n_reports_what_was_graded_not_what_was_attempted() -> None:
    """A score of 0.25 over 89 questions and a score of 0.25 over 150 are different claims. The
    second one would be a lie about how much evidence there is."""
    samples = [_sample(f"smoke:dev:{i}", ("calculation",)) for i in range(10)]
    results = [
        _result(f"smoke:dev:{i}", "exact_match", True if i < 2 else (False if i < 4 else None))
        for i in range(10)
    ]

    rollup = rollup_capabilities(samples, results)
    assert rollup[CapabilityDimension.NUMERICAL_ACCURACY].n == 4
    assert rollup[CapabilityDimension.NUMERICAL_ACCURACY].mean == 0.5


def test_a_dimension_with_nothing_gradable_is_absent_rather_than_zero() -> None:
    """`0.0` reads as "measured, and hopeless". Absence reads as "not measured". Only one of those
    is true when every question in the dimension was ungradeable."""
    samples = [_sample(f"smoke:dev:{i}", ("calculation",)) for i in range(3)]
    results = [_result(f"smoke:dev:{i}", "exact_match", None) for i in range(3)]

    rollup = rollup_capabilities(samples, results)
    assert CapabilityDimension.NUMERICAL_ACCURACY not in rollup


def test_a_genuine_zero_is_still_a_zero() -> None:
    """The fix must not swallow real failures. `passed=False` is a measured failure and it counts."""
    samples = [_sample(f"smoke:dev:{i}", ("calculation",)) for i in range(2)]
    results = [_result(f"smoke:dev:{i}", "exact_match", False) for i in range(2)]

    rollup = rollup_capabilities(samples, results)
    assert rollup[CapabilityDimension.NUMERICAL_ACCURACY].mean == 0.0
    assert rollup[CapabilityDimension.NUMERICAL_ACCURACY].n == 2


# --------------------------------------------------------------------------- the right metric


def test_the_refusal_dimension_is_scored_by_the_refusal_metric() -> None:
    """The contradiction that reached a real report: refusal_correctness 1.000, and a
    calibration-and-refusal capability of 0.0, in the same file.

    Accuracy cannot grade an unanswerable question — there is no number to get right — so feeding it
    to the refusal dimension scores that dimension on a metric that does not apply to it. Here the
    model refuses correctly every time and cannot compute anything; the refusal dimension must say
    1.0, because refusing is what it is being asked about.
    """
    samples = [_sample(f"smoke:dev:{i}", ("calibration_refusal",)) for i in range(3)]
    preferred = [_result(f"smoke:dev:{i}", "smb_cfo_accuracy", None) for i in range(3)]
    everything = [
        *preferred,
        *[_result(f"smoke:dev:{i}", "smb_cfo_refusal_correctness", True) for i in range(3)],
    ]

    rollup = rollup_capabilities(samples, preferred, all_results=everything)
    refusal = rollup[CapabilityDimension.CALIBRATION_AND_REFUSAL]

    assert refusal.mean == 1.0
    assert refusal.n == 3


def test_a_model_that_fails_to_refuse_still_scores_zero_on_the_refusal_dimension() -> None:
    """The fix must not make the dimension unable to report bad news — which is the whole reason it
    exists."""
    samples = [_sample(f"smoke:dev:{i}", ("calibration_refusal",)) for i in range(3)]
    preferred = [_result(f"smoke:dev:{i}", "smb_cfo_accuracy", None) for i in range(3)]
    everything = [
        *preferred,
        *[_result(f"smoke:dev:{i}", "smb_cfo_refusal_correctness", False) for i in range(3)],
    ]

    rollup = rollup_capabilities(samples, preferred, all_results=everything)
    assert rollup[CapabilityDimension.CALIBRATION_AND_REFUSAL].mean == 0.0


def test_other_dimensions_still_use_the_benchmarks_preferred_metric() -> None:
    """Only the dimensions that a *different* metric measures are overridden. Everything else keeps
    the headline metric it always had."""
    samples = [_sample("smoke:dev:0", ("calculation",))]
    preferred = [_result("smoke:dev:0", "finqa_answer_accuracy", True)]
    everything = [*preferred, _result("smoke:dev:0", "smb_cfo_refusal_correctness", False)]

    rollup = rollup_capabilities(samples, preferred, all_results=everything)
    assert rollup[CapabilityDimension.NUMERICAL_ACCURACY].mean == 1.0


# --------------------------------------------------------------------------- against the real data


def test_the_real_financebench_split_has_ungradeable_questions_to_exclude() -> None:
    """Guards the guard. If FinanceBench had no analytical questions, the bug above would have been
    invisible and these tests would be theatre.

    150 questions; 89 have a deterministically checkable answer (52 numeric + 37 boolean); the other
    61 are analytical. Those 61 were being scored as zeros.
    """
    from financebench.datasets.financebench import FinanceBenchAdapter

    samples = FinanceBenchAdapter().load("open_source")
    shapes = [s.gold.answer_type.value for s in samples]
    assert len(samples) == 150
    assert sum(1 for s in shapes if s in ("numeric", "boolean")) == 89
    assert sum(1 for s in shapes if s == "text") == 61


@pytest.mark.parametrize("dimension", list(CapabilityDimension))
def test_every_dimension_survives_an_empty_rollup(dimension: CapabilityDimension) -> None:
    assert rollup_capabilities([], []) == {}
