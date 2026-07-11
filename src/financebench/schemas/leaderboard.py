"""The leaderboard record schema — one row per (model, run) in the global leaderboard, kept
separate from a single run's own artifacts so the leaderboard builder can validate compatibility
before comparing two rows.

Two rows are only comparable if their :class:`RunFingerprint` matches. A leaderboard that silently
mixes a ``context_given`` run with a ``retrieval_required`` one, or a 40-sample subset with a
full-split run, is worse than no leaderboard — it invents a ranking out of an artifact of the
configuration. So the fingerprint travels with the record, and the builder groups by it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from financebench.schemas.common import EvalMode, RunType

__all__ = ["LeaderboardRecord", "RunFingerprint", "sample_id_set_hash"]


def sample_id_set_hash(sample_ids: Iterable[str]) -> str:
    """A stable hash of the exact set of samples a run evaluated.

    Sorted, so evaluation order can't change it; truncated, because it is an identity token, not a
    security primitive.
    """
    joined = "\n".join(sorted(set(sample_ids)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


class RunFingerprint(BaseModel):
    """Everything that must match before two runs may be compared or ranked against each other.

    Deliberately includes ``sample_id_set``: two runs over "finqa test" are not comparable if one
    used ``--max-samples 40`` and the other used all 1,147, however tempting the arithmetic.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_or_group: str
    benchmark_versions: tuple[str, ...] = Field(default_factory=tuple)
    prompt_profile: str
    eval_mode: EvalMode = EvalMode.CONTEXT_GIVEN
    evaluator_version: str
    sample_id_set: str

    def comparable_with(self, other: RunFingerprint) -> bool:
        return self == other


class LeaderboardRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    model_ref: str
    provider: str
    #: ``mock_test`` runs are produced by a simulator holding the answer key. They are recorded
    #: (so a reader can see the pipeline was exercised) but never ranked.
    run_type: RunType = RunType.REAL
    eligible_for_leaderboard: bool = True
    eval_mode: EvalMode = EvalMode.CONTEXT_GIVEN
    fingerprint: RunFingerprint | None = None
    fci: float | None = None
    band: str | None = None
    verdict: str | None = None
    provisional: bool = True
    critical_gate_failed: bool = False
    capability_scores: dict[str, float] = Field(default_factory=dict)
    native_scores: dict[str, float] = Field(default_factory=dict)
    coverage_summary: dict[str, Any] = Field(default_factory=dict)
    deployment_efficiency: dict[str, Any] | None = None
    created_at: str
