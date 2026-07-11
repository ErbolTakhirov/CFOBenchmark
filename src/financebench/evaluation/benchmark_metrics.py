"""Which metrics apply to which run.

A metric is chosen by **(benchmark, prompt profile)**, not by benchmark alone. That is not
bookkeeping — it is the difference between an honest number and a fabricated one:

FinQA's official metrics (execution accuracy, program accuracy) are defined over a *predicted
program*. If the run asked the model for a plain number, there is no program to execute or
compare, and reporting "program accuracy: 0.0" would be a lie — the model was never asked. So a
direct-answer run gets ``finqa_answer_accuracy`` (ours, tolerance-based, named so it cannot be
mistaken for the official metric), and a ``program_v1`` run gets the two official ones.

``metric_details.jsonl`` records every applicable metric per sample; only the *preferred* one feeds
the capability-dimension rollup.
"""

from __future__ import annotations

from financebench.evaluation.metrics.base import Metric, create_metric
from financebench.prompts.profiles import create_prompt_profile
from financebench.schemas.common import DEFAULT_PROMPT_PROFILE

__all__ = ["metrics_for_run", "preferred_metric_name"]

#: (benchmark, elicits_program) -> the metric that feeds the capability rollup.
_PREFERRED: dict[tuple[str, bool], str] = {
    ("finqa", True): "finqa_execution_accuracy",  # official
    ("finqa", False): "finqa_answer_accuracy",  # ours; NOT the official metric
    ("tatqa", False): "tatqa_exact_match",  # official
    ("tatqa", True): "tatqa_exact_match",  # TAT-QA has no program mode
    ("finance_reasoning", False): "finance_reasoning_accuracy",  # official
    ("finance_reasoning", True): "finance_reasoning_accuracy",
    ("financebench", False): "financebench_answer_accuracy",  # OURS — FinanceBench has no evaluator
    ("financebench", True): "financebench_answer_accuracy",
}

#: (benchmark, elicits_program) -> further metrics worth recording, beyond the preferred one.
_ADDITIONAL: dict[tuple[str, bool], tuple[str, ...]] = {
    ("finqa", True): ("finqa_program_accuracy",),
    ("tatqa", False): ("tatqa_f1", "tatqa_scale_accuracy"),
    ("tatqa", True): ("tatqa_f1", "tatqa_scale_accuracy"),
    # The hallucination detector applies to ALL 150 regardless of gold answer shape, and is the
    # single most important number this benchmark produces.
    ("financebench", False): (
        "financebench_unsupported_numeric_claim",
        "financebench_citation_accuracy",
    ),
    ("financebench", True): (
        "financebench_unsupported_numeric_claim",
        "financebench_citation_accuracy",
    ),
}


def _elicits_program(prompt_profile: str) -> bool:
    try:
        return create_prompt_profile(prompt_profile).elicits_program
    except Exception:
        return False


def preferred_metric_name(benchmark: str, prompt_profile: str = DEFAULT_PROMPT_PROFILE) -> str:
    """The metric whose result feeds the capability-dimension rollup for this run."""
    key = (benchmark, _elicits_program(prompt_profile))
    return _PREFERRED.get(key, "exact_match")


def metrics_for_run(
    benchmark: str, prompt_profile: str = DEFAULT_PROMPT_PROFILE
) -> tuple[Metric, ...]:
    """Every metric to compute and report for a sample from ``benchmark`` under this profile."""
    key = (benchmark, _elicits_program(prompt_profile))
    names = {
        "exact_match",
        preferred_metric_name(benchmark, prompt_profile),
        *_ADDITIONAL.get(key, ()),
    }
    return tuple(create_metric(name) for name in sorted(names))
