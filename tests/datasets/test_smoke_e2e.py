"""End-to-end proof that the `smoke` adapter's `fully_supported` claim is real: it loads, every
record validates against the canonical schema, and running it through the actual engine against
the mock provider produces correct, gradable answers. Every dataset manifest claiming
`fully_supported` or `supported_public_subset` must have a matching file here (or under
`tests/datasets/`) — enforced by the repo-hygiene test in `tests/test_manifest_hygiene.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from financebench.datasets.smoke.adapter import SmokeDatasetAdapter
from financebench.execution.cache import ResponseCache
from financebench.execution.engine import RunEngine
from financebench.models.mock import MockProvider, build_mock_oracle
from financebench.schemas.manifest import AdapterStatus
from financebench.schemas.model_io import ModelSpec
from financebench.schemas.run import RunConfig
from financebench.utils.errors import DatasetLoadError


def test_manifest_declares_fully_supported_and_is_tested() -> None:
    manifest = SmokeDatasetAdapter().manifest()
    assert manifest.status is AdapterStatus.FULLY_SUPPORTED
    assert manifest.status_tested_at is not None
    assert manifest.local_splits == ("dev",)


def test_load_returns_ten_samples_all_in_the_dev_split() -> None:
    samples = SmokeDatasetAdapter().load("dev")
    assert len(samples) == 10
    assert {s.split for s in samples} == {"dev"}
    assert {s.benchmark for s in samples} == {"smoke"}


def test_sample_ids_are_unique() -> None:
    samples = SmokeDatasetAdapter().load("dev")
    assert len({s.sample_id for s in samples}) == len(samples)


def test_unknown_split_raises_dataset_load_error() -> None:
    with pytest.raises(DatasetLoadError, match="no split"):
        SmokeDatasetAdapter().load("test")


@pytest.mark.asyncio
async def test_smoke_samples_score_perfectly_against_the_echo_gold_mock(tmp_path: Path) -> None:
    """The actual end-to-end run the `fully_supported` claim rests on."""
    samples = SmokeDatasetAdapter().load("dev")
    result = await RunEngine().run(
        samples=samples,
        model=ModelSpec.parse("mock/echo-gold"),
        config=RunConfig(),
        cache=ResponseCache(tmp_path),
        provider=MockProvider(oracle=build_mock_oracle(samples)),
    )
    assert result.n_samples == 10
    assert result.n_errors == 0
    for sample, prediction in zip(samples, result.predictions, strict=True):
        assert prediction.response is not None
        answer = prediction.response.financial_answer
        assert answer is not None
        assert answer.answer == sample.gold.answer
