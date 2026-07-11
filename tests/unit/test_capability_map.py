from __future__ import annotations

from tests.factories import make_sample

from financebench.evaluation.capability_map import (
    CAPABILITY_WEIGHTS,
    CapabilityDimension,
    dimensions_for_sample,
    rollup_capabilities,
)
from financebench.schemas.metric import MetricResult


def _result(sample_id: str, passed: bool) -> MetricResult:
    return MetricResult(sample_id=sample_id, metric_name="exact_match", value=passed, passed=passed)


def test_every_dimension_has_a_weight_and_they_sum_to_one() -> None:
    assert set(CAPABILITY_WEIGHTS) == set(CapabilityDimension)
    assert abs(sum(CAPABILITY_WEIGHTS.values()) - 1.0) < 1e-9


def test_tags_map_to_their_dimensions() -> None:
    sample = make_sample(capability_tags=("calculation",))
    assert dimensions_for_sample(sample) == (CapabilityDimension.NUMERICAL_ACCURACY,)


def test_a_sample_can_count_toward_more_than_one_dimension() -> None:
    """A cited table lookup is both a grounding task and a table/text task. Forcing it into one
    would understate whichever it was denied."""
    sample = make_sample(capability_tags=("table_text", "evidence_grounding"))
    assert dimensions_for_sample(sample) == (
        CapabilityDimension.DOCUMENT_GROUNDING,
        CapabilityDimension.TABLE_TEXT_REASONING,
    )


def test_two_tags_mapping_to_one_dimension_are_deduped() -> None:
    sample = make_sample(capability_tags=("analysis", "insight"))
    assert dimensions_for_sample(sample) == (CapabilityDimension.ANALYTICAL_INSIGHT,)


def test_an_unmapped_tag_maps_to_nothing_rather_than_being_guessed() -> None:
    """An unrecognized tag is a configuration gap to notice, not something to default away."""
    sample = make_sample(capability_tags=("some-tag-nobody-defined",))
    assert dimensions_for_sample(sample) == ()


def test_rollup_produces_one_aggregate_per_dimension() -> None:
    samples = [
        make_sample(sample_id="smoke:dev:1", capability_tags=("calculation",)),
        make_sample(sample_id="smoke:dev:2", capability_tags=("calculation",)),
    ]
    results = [_result("smoke:dev:1", True), _result("smoke:dev:2", False)]

    rolled = rollup_capabilities(samples, results)
    aggregate = rolled[CapabilityDimension.NUMERICAL_ACCURACY]
    assert aggregate.n == 2
    assert aggregate.mean == 0.5


def test_a_bigger_benchmark_does_not_get_a_bigger_vote() -> None:
    """The whole reason for macro-averaging.

    Benchmark "big" contributes 100 samples, all wrong. Benchmark "small" contributes 2, both
    right. A micro-average (pooling all 102) gives 0.02 — i.e. the score becomes "whatever the
    biggest dataset says". The macro-average gives 0.5, because each benchmark gets one vote.
    Dataset size is an artifact of how the data was collected, not a claim about what matters.
    """
    samples = []
    results = []
    for i in range(100):
        sample_id = f"big:dev:{i}"
        samples.append(
            make_sample(
                sample_id=sample_id,
                benchmark="big",
                task_family="t",
                capability_tags=("calculation",),
            )
        )
        results.append(_result(sample_id, False))
    for i in range(2):
        sample_id = f"small:dev:{i}"
        samples.append(
            make_sample(
                sample_id=sample_id,
                benchmark="small",
                task_family="t",
                capability_tags=("calculation",),
            )
        )
        results.append(_result(sample_id, True))

    rolled = rollup_capabilities(samples, results)
    aggregate = rolled[CapabilityDimension.NUMERICAL_ACCURACY]

    micro_average = 2 / 102
    assert aggregate.mean == 0.5, "each benchmark gets one vote, regardless of its size"
    assert aggregate.mean != micro_average
    assert aggregate.n == 102, "but n still reports how much evidence there actually is"


def test_a_sample_with_no_matching_result_is_simply_excluded() -> None:
    samples = [make_sample(sample_id="smoke:dev:1", capability_tags=("calculation",))]
    assert rollup_capabilities(samples, []) == {}
