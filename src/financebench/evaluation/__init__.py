"""Evaluation: metrics (native + unified) and the capability-dimension rollup.

The full evaluator orchestration (native per-benchmark scoring, the LLM-judge framework, FCI/
gates/confidence-intervals) is built out across Milestones 2-6; this package currently holds the
Milestone 1 foundation both of those build on.
"""

from __future__ import annotations

from financebench.evaluation import (
    grounding as _grounding,  # noqa: F401  (registers grounding metrics)
)
from financebench.evaluation import native as _native  # noqa: F401  (registers native metrics)
from financebench.evaluation.benchmark_metrics import metrics_for_run, preferred_metric_name
from financebench.evaluation.capability_map import (
    CAPABILITY_WEIGHTS,
    CapabilityDimension,
    dimensions_for_sample,
    rollup_capabilities,
)

__all__ = [
    "CAPABILITY_WEIGHTS",
    "CapabilityDimension",
    "dimensions_for_sample",
    "metrics_for_run",
    "preferred_metric_name",
    "rollup_capabilities",
]
