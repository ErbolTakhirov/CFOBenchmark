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
from financebench.schemas.common import DEFAULT_PROMPT_PROFILE, EvalMode

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
    ("smb_cfo", False): "smb_cfo_accuracy",  # gold from a Python oracle, never an LLM
    ("smb_cfo", True): "smb_cfo_accuracy",
    # ConvFinQA's official metrics grade a *program*. A direct-answer run has none, so it gets ours
    # — under a name that cannot be mistaken for the official one.
    ("convfinqa", True): "convfinqa_execution_accuracy",  # official
    ("convfinqa", False): "convfinqa_turn_accuracy",  # ours; NOT the official metric
    # SECQUE has NO official metric — its gold is an expert's prose. The "preferred" one here is the
    # hallucination detector, deliberately: it is the only deterministic check a fluent answer cannot
    # talk its way past, and it is the one whose failure actually matters. Analytical QUALITY is the
    # judge's job (evaluation/judge/), reported separately and never folded into this.
    ("secque", False): "secque_unsupported_numeric_claim",
    ("secque", True): "secque_unsupported_numeric_claim",
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
    # Refusal correctness and injection resistance measure things accuracy cannot see: whether the
    # model knows when it CANNOT answer, and whether it can be talked into lying by its own data.
    ("smb_cfo", False): ("smb_cfo_refusal_correctness", "smb_cfo_injection_resistance"),
    ("smb_cfo", True): ("smb_cfo_refusal_correctness", "smb_cfo_injection_resistance"),
    # Program accuracy sees what execution accuracy cannot: `(a - b) / b` and `a / b - 1` agree on
    # the number and disagree on the reasoning.
    ("convfinqa", True): ("convfinqa_program_accuracy", "convfinqa_turn_accuracy"),
    ("secque", False): (
        "secque_numeric_agreement",
        "secque_comparison_direction",
        "secque_filing_identification",
    ),
    ("secque", True): (
        "secque_numeric_agreement",
        "secque_comparison_direction",
        "secque_filing_identification",
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


#: Computed for ANY benchmark run in tool_assisted mode.
#:
#: Keyed on the eval MODE, not the benchmark — which is the whole point. "Can this model use a
#: calculator" is a property of the model, not of FinQA, and the same four metrics apply whether the
#: arithmetic came from a 10-K or a small business's ledger. The first version of this file keyed
#: only on the benchmark, so a live tool-assisted run completed happily and computed NONE of them.
_TOOL_METRICS: tuple[str, ...] = (
    "tool_selection_accuracy",
    "tool_execution_success",
    "tool_result_utilization",  # the one that matters
    "tool_security_rejection",
    # Each of these separates a failure the others conflate. "Called nothing" and "invented a tool"
    # both show up as tool_selection_accuracy=0, and they have opposite fixes.
    "tool_invocation_rate",
    "tool_argument_validity",
    "tool_hallucination_rate",
    "tool_error_recovery",
)


def metrics_for_run(
    benchmark: str,
    prompt_profile: str = DEFAULT_PROMPT_PROFILE,
    eval_mode: EvalMode = EvalMode.CONTEXT_GIVEN,
) -> tuple[Metric, ...]:
    """Every metric to compute and report for a sample from ``benchmark`` under this profile/mode."""
    key = (benchmark, _elicits_program(prompt_profile))
    names = {
        "exact_match",
        preferred_metric_name(benchmark, prompt_profile),
        *_ADDITIONAL.get(key, ()),
    }
    if eval_mode is EvalMode.TOOL_ASSISTED:
        # The benchmark's own accuracy metric still applies — the answer is still right or wrong.
        # These say WHERE it broke, which accuracy cannot.
        names.update(_TOOL_METRICS)
    return tuple(create_metric(name) for name in sorted(names))
