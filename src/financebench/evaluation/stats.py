"""Statistical honesty: confidence intervals, and paired comparison of two models.

A benchmark run on 40 samples is a small sample, and 45 % vs 50 % on 40 samples is almost certainly
noise. Reporting a ranking from that would be inventing a result. So:

- every score carries a **bootstrap 95 % confidence interval**;
- two models are compared **paired on identical sample IDs**, never as two independent samples —
  the same questions are hard for both, and pairing removes that shared variance instead of
  letting it swamp the difference;
- a difference is only called a difference when the paired interval excludes zero.

Deterministic by construction: the bootstrap is seeded, so the same run always yields the same
interval. A confidence interval that wobbles between invocations is not a confidence interval.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "MIN_SAMPLES_FOR_A_CLAIM",
    "BootstrapResult",
    "PairedComparison",
    "bootstrap_ci",
    "paired_bootstrap",
]

#: Below this, no ranking claim is made at all — only the raw number, with a warning attached.
MIN_SAMPLES_FOR_A_CLAIM = 30

_DEFAULT_ITERATIONS = 2000
_DEFAULT_SEED = 42


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    n: int
    #: True when n is too small for the interval to mean much. Reported, never silently ignored.
    underpowered: bool

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    iterations: int = _DEFAULT_ITERATIONS,
    seed: int = _DEFAULT_SEED,
) -> BootstrapResult | None:
    """A percentile bootstrap confidence interval for the mean. ``None`` for an empty sample."""
    if not values:
        return None
    n = len(values)
    mean = sum(values) / n

    if n == 1:
        return BootstrapResult(mean=mean, ci_low=mean, ci_high=mean, n=1, underpowered=True)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()

    alpha = (1.0 - confidence) / 2.0
    low = means[int(alpha * iterations)]
    high = means[min(int((1.0 - alpha) * iterations), iterations - 1)]
    return BootstrapResult(
        mean=mean, ci_low=low, ci_high=high, n=n, underpowered=n < MIN_SAMPLES_FOR_A_CLAIM
    )


@dataclass(frozen=True)
class PairedComparison:
    """Two models on the *same* questions."""

    n_paired: int
    mean_a: float
    mean_b: float
    mean_difference: float
    ci_low: float
    ci_high: float
    #: Only true when the interval excludes zero AND there are enough pairs to say so.
    significant: bool
    underpowered: bool
    #: Where they actually differ — the interesting part, and what a mean hides.
    a_right_b_wrong: int
    b_right_a_wrong: int
    both_right: int
    both_wrong: int

    def verdict(self, name_a: str, name_b: str) -> str:
        if self.underpowered:
            return (
                f"Too few paired samples ({self.n_paired} < {MIN_SAMPLES_FOR_A_CLAIM}) to claim a "
                f"difference between {name_a} and {name_b}."
            )
        if not self.significant:
            return (
                f"No significant difference between {name_a} and {name_b} "
                f"(difference {self.mean_difference:+.3f}, 95% CI "
                f"[{self.ci_low:+.3f}, {self.ci_high:+.3f}] — includes zero)."
            )
        better, worse = (name_a, name_b) if self.mean_difference > 0 else (name_b, name_a)
        return (
            f"{better} beats {worse} (difference {abs(self.mean_difference):.3f}, 95% CI "
            f"[{self.ci_low:+.3f}, {self.ci_high:+.3f}] — excludes zero)."
        )


def paired_bootstrap(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    *,
    confidence: float = 0.95,
    iterations: int = _DEFAULT_ITERATIONS,
    seed: int = _DEFAULT_SEED,
) -> PairedComparison | None:
    """Compare two models on the samples they *both* answered.

    Pairing is the point. Two models scoring 45 % and 50 % on 40 questions look
    indistinguishable as independent samples — but if the second got every question the first got,
    plus two more, that is a real (if small) difference, and an unpaired test would miss it. And in
    the other direction, an unpaired test can easily manufacture a difference out of nothing.
    """
    shared = sorted(set(scores_a) & set(scores_b))
    if not shared:
        return None

    differences = [scores_a[sample_id] - scores_b[sample_id] for sample_id in shared]
    n = len(shared)
    mean_difference = sum(differences) / n

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        resample = [differences[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()

    alpha = (1.0 - confidence) / 2.0
    low = means[int(alpha * iterations)]
    high = means[min(int((1.0 - alpha) * iterations), iterations - 1)]

    underpowered = n < MIN_SAMPLES_FOR_A_CLAIM
    significant = (low > 0 or high < 0) and not underpowered

    return PairedComparison(
        n_paired=n,
        mean_a=sum(scores_a[s] for s in shared) / n,
        mean_b=sum(scores_b[s] for s in shared) / n,
        mean_difference=mean_difference,
        ci_low=low,
        ci_high=high,
        significant=significant,
        underpowered=underpowered,
        a_right_b_wrong=sum(1 for s in shared if scores_a[s] > scores_b[s]),
        b_right_a_wrong=sum(1 for s in shared if scores_b[s] > scores_a[s]),
        both_right=sum(1 for s in shared if scores_a[s] > 0 and scores_b[s] > 0),
        both_wrong=sum(1 for s in shared if scores_a[s] == 0 and scores_b[s] == 0),
    )
