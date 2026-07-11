"""The ten capability dimensions, and how a sample's score reaches one.

Two design decisions here matter more than the list itself.

**Macro-averaging, not micro-averaging.** A score is rolled up
sample → task family → benchmark → capability, averaging at each level. If you instead pool every
sample and take one mean, then whichever benchmark happens to have the most rows decides the
capability score — FinanceReasoning's 2,238 questions would drown FinQA's 1,147, and a capability
would silently become "whatever the biggest dataset measures". Dataset size is an artifact of how
the data was collected; it is not a statement about what matters.

**An unmapped tag maps to nothing.** A sample whose ``capability_tags`` don't match any dimension
is excluded from every capability rollup rather than being guessed into one. An unmapped tag is a
configuration gap to notice and fix, not something to paper over with a default.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from enum import StrEnum

from financebench.evaluation.metrics.base import aggregate_metric
from financebench.schemas.metric import MetricAggregate, MetricResult
from financebench.schemas.sample import CanonicalSample

__all__ = [
    "CAPABILITY_WEIGHTS",
    "CapabilityDimension",
    "dimensions_for_sample",
    "macro_average",
    "rollup_capabilities",
]


class CapabilityDimension(StrEnum):
    """The ten dimensions scored independently, so a model cannot hide a critical weakness behind
    one strong area."""

    NUMERICAL_ACCURACY = "numerical_accuracy"
    FINANCIAL_FORMULA_REASONING = "financial_formula_reasoning"
    TABLE_TEXT_REASONING = "table_text_reasoning"
    DOCUMENT_GROUNDING = "document_grounding"
    RETRIEVAL_QUALITY = "retrieval_quality"
    ANALYTICAL_INSIGHT = "analytical_insight"
    CONVERSATION_CONSISTENCY = "conversation_consistency"
    CALIBRATION_AND_REFUSAL = "calibration_and_refusal"
    BILINGUAL_EN_RU = "bilingual_en_ru"
    TOOL_USE_RELIABILITY = "tool_use_reliability"


#: Weights for the Finance Capability Index. Numerical accuracy dominates because in finance a
#: wrong number is not a partially-right answer — it is a wrong answer with a plausible shape.
CAPABILITY_WEIGHTS: dict[CapabilityDimension, float] = {
    CapabilityDimension.NUMERICAL_ACCURACY: 0.20,
    CapabilityDimension.FINANCIAL_FORMULA_REASONING: 0.15,
    CapabilityDimension.TABLE_TEXT_REASONING: 0.12,
    CapabilityDimension.DOCUMENT_GROUNDING: 0.12,
    CapabilityDimension.RETRIEVAL_QUALITY: 0.08,
    CapabilityDimension.ANALYTICAL_INSIGHT: 0.10,
    CapabilityDimension.CONVERSATION_CONSISTENCY: 0.07,
    CapabilityDimension.CALIBRATION_AND_REFUSAL: 0.10,
    CapabilityDimension.BILINGUAL_EN_RU: 0.03,
    CapabilityDimension.TOOL_USE_RELIABILITY: 0.03,
}

assert abs(sum(CAPABILITY_WEIGHTS.values()) - 1.0) < 1e-9, "capability weights must sum to 1.0"
assert set(CAPABILITY_WEIGHTS) == set(CapabilityDimension), "every dimension needs a weight"


_TAG_TO_DIMENSION: dict[str, CapabilityDimension] = {
    "calculation": CapabilityDimension.NUMERICAL_ACCURACY,
    "formula": CapabilityDimension.FINANCIAL_FORMULA_REASONING,
    "ratio": CapabilityDimension.FINANCIAL_FORMULA_REASONING,
    "table_text": CapabilityDimension.TABLE_TEXT_REASONING,
    "evidence_grounding": CapabilityDimension.DOCUMENT_GROUNDING,
    "retrieval": CapabilityDimension.RETRIEVAL_QUALITY,
    "analysis": CapabilityDimension.ANALYTICAL_INSIGHT,
    "insight": CapabilityDimension.ANALYTICAL_INSIGHT,
    "conversation": CapabilityDimension.CONVERSATION_CONSISTENCY,
    "calibration_refusal": CapabilityDimension.CALIBRATION_AND_REFUSAL,
    "bilingual": CapabilityDimension.BILINGUAL_EN_RU,
    "tool_use": CapabilityDimension.TOOL_USE_RELIABILITY,
}


def dimensions_for_sample(sample: CanonicalSample) -> tuple[CapabilityDimension, ...]:
    """Which dimension(s) a sample's score counts toward, from its ``capability_tags``.

    A sample with no recognized tag maps to nothing and is excluded from every rollup, rather than
    being guessed into a dimension it may not belong in.
    """
    matched = {_TAG_TO_DIMENSION[tag] for tag in sample.capability_tags if tag in _TAG_TO_DIMENSION}
    return tuple(sorted(matched, key=lambda dimension: dimension.value))


def macro_average(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _score(result: MetricResult) -> float:
    if isinstance(result.value, bool):
        return 1.0 if result.value else 0.0
    if isinstance(result.value, int | float):
        return float(result.value)
    return 0.0


def rollup_capabilities(
    samples: Sequence[CanonicalSample], results: Sequence[MetricResult]
) -> dict[CapabilityDimension, MetricAggregate]:
    """Roll up per-sample results into one aggregate per capability, **macro-averaging** at each
    level: sample → task family → benchmark → capability.

    Each level's mean is taken over the level below, so a benchmark with ten times the rows does
    not get ten times the say. ``n`` on the returned aggregate is still the true sample count, so a
    reader can see how much evidence a score rests on.
    """
    by_sample_id = {result.sample_id: result for result in results}

    # capability -> benchmark -> task_family -> [scores]
    buckets: dict[CapabilityDimension, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    counts: dict[CapabilityDimension, int] = defaultdict(int)
    per_dimension_results: dict[CapabilityDimension, list[MetricResult]] = defaultdict(list)

    for sample in samples:
        result = by_sample_id.get(sample.sample_id)
        if result is None:
            continue
        for dimension in dimensions_for_sample(sample):
            buckets[dimension][sample.benchmark][sample.task_family].append(_score(result))
            counts[dimension] += 1
            per_dimension_results[dimension].append(result)

    aggregates: dict[CapabilityDimension, MetricAggregate] = {}
    for dimension, by_benchmark in buckets.items():
        benchmark_means: list[float] = []
        for by_task in by_benchmark.values():
            task_means = [
                mean for scores in by_task.values() if (mean := macro_average(scores)) is not None
            ]
            benchmark_mean = macro_average(task_means)
            if benchmark_mean is not None:
                benchmark_means.append(benchmark_mean)

        mean = macro_average(benchmark_means)
        # Reuse the shared aggregator for shape, then override the mean with the macro-average.
        base = aggregate_metric(dimension.value, per_dimension_results[dimension])
        aggregates[dimension] = base.model_copy(update={"mean": mean, "n": counts[dimension]})

    return aggregates
