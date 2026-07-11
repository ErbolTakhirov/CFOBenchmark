"""The fingerprint must actually cover the pipeline it claims to fingerprint.

The evaluator fingerprint exists so that a change to *our* code — a parser fix, a metric correction,
a regenerated dataset — cannot silently move a score and have the new number sit on a leaderboard
next to the old one as though they were comparable.

It only works if it is complete. And it was not: ``financebench`` and ``smb_cfo`` were both missing
from ``DATASET_ADAPTER_VERSIONS``, and every one of their metrics was missing from
``METRIC_VERSIONS``. Nothing failed. The digest was computed, written into ``environment.json``, and
compared between runs — while being **blind to two entire benchmarks**. Regenerating the SMB-CFO
oracles would have changed every SMB-CFO score in the repo and left the fingerprint identical, which
is precisely the failure the fingerprint was built to prevent, wearing the fingerprint's own badge.

A fingerprint with a hole in it is worse than no fingerprint, because it is *trusted*. So the
registries are the source of truth, and this asserts they agree: register a metric or an adapter, and
it must be versioned, or the suite fails.
"""

from __future__ import annotations

import financebench.datasets
import financebench.evaluation.native  # noqa: F401  (registers every native metric)
from financebench.datasets.base import available_datasets, get_dataset_class
from financebench.evaluation.fingerprint import (
    DATASET_ADAPTER_VERSIONS,
    METRIC_VERSIONS,
    current_fingerprint,
)
from financebench.evaluation.metrics.base import available_metrics, create_metric


def _shipped_datasets() -> set[str]:
    """Adapters that ship with the package.

    Other tests register throwaway adapters into the same global registry to exercise the
    registration machinery, and those must not be demanded of the fingerprint — a fixture invented
    inside a test function has no upstream commit to pin. Filtering by defining module keeps this
    test about the real ones without weakening it: add a real adapter and it appears here
    automatically.
    """
    return {
        name
        for name in available_datasets()
        if get_dataset_class(name).__module__.startswith("financebench.")
    }


def _shipped_metrics() -> set[str]:
    return {
        name
        for name in available_metrics()
        if type(create_metric(name)).__module__.startswith("financebench.")
    }


def test_every_registered_dataset_is_versioned_in_the_fingerprint() -> None:
    """A dataset whose adapter is not in the fingerprint can be regenerated, re-pinned to a new
    upstream commit, or have its parsing changed — and every run before and after will claim to be
    comparable."""
    missing = _shipped_datasets() - set(DATASET_ADAPTER_VERSIONS)
    assert not missing, (
        f"registered but unversioned: {sorted(missing)}. Add each to DATASET_ADAPTER_VERSIONS "
        "with the upstream commit (or generator version) its data comes from — otherwise its "
        "scores are being compared across changes the fingerprint cannot see."
    )


def test_every_registered_metric_is_versioned_in_the_fingerprint() -> None:
    """A metric whose behaviour changes moves every score computed with it. That is exactly what
    happened to ``smb_cfo_refusal_correctness``: v1 read a flag, v2 reads the answer, and on
    identical cached responses the score went from 0.667 to 1.000. Without a version bump those two
    numbers would have been presented as the same measurement."""
    missing = _shipped_metrics() - set(METRIC_VERSIONS)
    assert not missing, (
        f"registered but unversioned: {sorted(missing)}. Add each to METRIC_VERSIONS. If its "
        "behaviour ever changes, bump it — a metric that moves a score without changing the "
        "fingerprint makes old and new runs falsely comparable."
    )


def test_the_fingerprint_does_not_version_things_that_do_not_exist() -> None:
    """The other direction. A stale entry for a deleted metric would keep the digest changing (or
    not changing) for reasons unrelated to anything the pipeline actually does."""
    stale_metrics = set(METRIC_VERSIONS) - _shipped_metrics()
    stale_datasets = set(DATASET_ADAPTER_VERSIONS) - _shipped_datasets()
    assert not stale_metrics, f"versioned but not registered: {sorted(stale_metrics)}"
    assert not stale_datasets, f"versioned but not registered: {sorted(stale_datasets)}"


def test_the_digest_moves_when_a_metric_version_moves() -> None:
    """The whole mechanism in one line: if bumping a version left the digest alone, the fingerprint
    would be decoration."""
    before = current_fingerprint().digest

    original = METRIC_VERSIONS["exact_match"]
    METRIC_VERSIONS["exact_match"] = f"{original}-changed"
    try:
        after = current_fingerprint().digest
    finally:
        METRIC_VERSIONS["exact_match"] = original

    assert before != after
    assert current_fingerprint().digest == before, "the change must not leak into other tests"


def test_two_identical_pipelines_are_comparable_and_a_changed_one_is_not() -> None:
    first = current_fingerprint()
    assert first.comparable_with(current_fingerprint())

    original = METRIC_VERSIONS["exact_match"]
    METRIC_VERSIONS["exact_match"] = f"{original}-changed"
    try:
        assert not first.comparable_with(current_fingerprint())
    finally:
        METRIC_VERSIONS["exact_match"] = original
