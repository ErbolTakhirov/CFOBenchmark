"""The top-level scores, and the Finance Capability Index.

Model ability, retrieval ability and agent ability are **not averaged into one number**. They are
three different things, and a single "financial score" that mixes them tells you nothing about any
of them — a RAG pipeline can fail because the retriever missed the page or because the model
misread it, and those have opposite fixes.

So a run reports whichever of these its eval mode earned:

- **Financial Core Score** — ``context_given`` only. The model's own financial reasoning.
- **Financial RAG Score** — ``retrieval_required`` only. The retrieval system.
- **Financial Agent Score** — ``tool_assisted`` only. Tool selection and use.

The **Finance Capability Index** is a weighted **geometric** mean of the capability dimensions,
scaled by a reliability penalty. Geometric, not arithmetic, because the arithmetic mean lets a
model trade a catastrophic weakness for an unrelated strength: 0.9 grounding and 0.1 numerical
accuracy averages to a respectable 0.5, which is a lie about a model that cannot do arithmetic.
The geometric mean of the same pair is 0.3, which is the truth.

The FCI is only computed when there is enough coverage, no critical gate has failed, and the run is
not a mock. Otherwise it is ``None`` — never a number with an asterisk.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from financebench.evaluation.capability_map import CAPABILITY_WEIGHTS, CapabilityDimension
from financebench.evaluation.failures import (
    CATASTROPHIC_FAILURES,
    FailureRecord,
    FailureType,
)
from financebench.schemas.common import EvalMode
from financebench.schemas.metric import MetricAggregate

__all__ = ["MIN_DIMENSIONS_FOR_FCI", "FinanceScores", "compute_scores", "reliability_penalty"]

#: An index built from one dimension is not an index. Below this, the FCI is withheld.
MIN_DIMENSIONS_FOR_FCI = 3

_EPSILON = 1e-6

#: Floor on the reliability multiplier. Even a maximally unreliable model keeps some credit for the
#: answers it got right — the penalty is a discount, not an annihilation.
_PENALTY_FLOOR = 0.65


def reliability_penalty(failures: list[FailureRecord], n_scored: int) -> float:
    """A multiplier in ``[0.65, 1.0]`` reflecting how *dangerously* a model fails, not how often.

    Weighted towards the failures you cannot live with: a model that is merely wrong is discounted
    far less than one that is confidently, catastrophically wrong.
    """
    if n_scored == 0:
        return 1.0

    def rate(types: frozenset[FailureType]) -> float:
        return sum(1 for f in failures if f.failure_type in types) / n_scored

    catastrophic = rate(CATASTROPHIC_FAILURES)
    unsupported = rate(
        frozenset({FailureType.UNSUPPORTED_NUMERIC_CLAIM, FailureType.UNSUPPORTED_NARRATIVE_CLAIM})
    )
    invalid = rate(frozenset({FailureType.INVALID_STRUCTURED_RESPONSE}))

    penalty = 1.0 - (0.50 * catastrophic + 0.30 * unsupported + 0.20 * invalid)
    return max(_PENALTY_FLOOR, min(1.0, penalty))


@dataclass(frozen=True)
class FinanceScores:
    """The top-level numbers for one run. ``None`` means "not measured", never "zero"."""

    eval_mode: EvalMode
    core_score: float | None = None
    rag_score: float | None = None
    agent_score: float | None = None
    multimodal_score: float | None = None
    fci: float | None = None
    fci_withheld_because: str | None = None
    reliability_penalty: float = 1.0

    def to_json(self) -> dict[str, object]:
        return {
            "eval_mode": self.eval_mode.value,
            "financial_core_score": self.core_score,
            "financial_rag_score": self.rag_score,
            "financial_agent_score": self.agent_score,
            "multimodal_finance_score": self.multimodal_score,
            "finance_capability_index": self.fci,
            "fci_withheld_because": self.fci_withheld_because,
            "reliability_penalty": round(self.reliability_penalty, 4),
        }


def _weighted_geometric_mean(scores: Mapping[CapabilityDimension, float]) -> float:
    """exp(Σ wᵢ·ln(max(sᵢ, ε))) / exp(Σ wᵢ) — renormalized over the dimensions actually present.

    Renormalization matters: without it, a run that only covers 4 of the 10 dimensions would be
    scored as if the other 6 were zero, and every partial run would look catastrophic.
    """
    total_weight = sum(CAPABILITY_WEIGHTS[dimension] for dimension in scores)
    if total_weight <= 0:
        return 0.0
    log_sum = sum(
        CAPABILITY_WEIGHTS[dimension] * math.log(max(score, _EPSILON))
        for dimension, score in scores.items()
    )
    return math.exp(log_sum / total_weight)


def compute_scores(
    *,
    eval_mode: EvalMode,
    capabilities: Mapping[CapabilityDimension, MetricAggregate],
    failures: list[FailureRecord],
    n_scored: int,
    any_critical_gate_failed: bool,
    is_mock: bool,
    has_multimodal_coverage: bool = False,
) -> FinanceScores:
    """Compute the top-level scores for a run."""
    present = {
        dimension: aggregate.mean
        for dimension, aggregate in capabilities.items()
        if aggregate.mean is not None
    }
    overall = sum(present.values()) / len(present) if present else None

    # The mode decides which top-level score this run is even entitled to report. A context_given
    # run has said nothing about a retriever, and must not imply that it has.
    core = overall if eval_mode is EvalMode.CONTEXT_GIVEN else None
    rag = overall if eval_mode is EvalMode.RETRIEVAL_REQUIRED else None
    agent = overall if eval_mode is EvalMode.TOOL_ASSISTED else None

    penalty = reliability_penalty(failures, n_scored)

    withheld: str | None = None
    fci: float | None = None
    if is_mock:
        withheld = "the mock provider was used — no model was evaluated"
    elif len(present) < MIN_DIMENSIONS_FOR_FCI:
        withheld = (
            f"only {len(present)} capability dimension(s) had coverage "
            f"(minimum {MIN_DIMENSIONS_FOR_FCI}); an index built from one dimension is not an index"
        )
    elif any_critical_gate_failed:
        withheld = (
            "a critical gate failed — a single index would let a strong average hide the kind of "
            "error that is not a near-miss in finance"
        )
    else:
        fci = round(_weighted_geometric_mean(present) * penalty, 4)

    return FinanceScores(
        eval_mode=eval_mode,
        core_score=core,
        rag_score=rag,
        agent_score=agent,
        multimodal_score=overall if has_multimodal_coverage else None,
        fci=fci,
        fci_withheld_because=withheld,
        reliability_penalty=penalty,
    )
