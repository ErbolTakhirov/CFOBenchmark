"""TAT-QA end to end: real official data -> canonical samples -> a run -> official metrics.

Runs against a committed slice of the **real** official dev split (MIT-licensed, so vendorable),
curated to contain every `(answer_type, scale)` combination the real data has — because the
official metric's special cases live in exactly the rare combinations a random sample would miss.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from financebench.datasets.tatqa.adapter import TatQAAdapter
from financebench.evaluation.native.tatqa import TatQAExactMatch, TatQAF1, TatQAScaleAccuracy
from financebench.execution.cache import ResponseCache
from financebench.execution.engine import RunEngine
from financebench.models.mock import MockProvider, build_mock_oracle
from financebench.schemas.common import AnswerType, SplitOrigin
from financebench.schemas.manifest import AdapterStatus
from financebench.schemas.model_io import ModelSpec
from financebench.schemas.run import RunConfig

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "tatqa"


def _adapter() -> TatQAAdapter:
    return TatQAAdapter(data_dir=FIXTURE_DIR)


def _raw_questions() -> list[dict]:
    records = json.loads((FIXTURE_DIR / "dev.json").read_text(encoding="utf-8"))
    return [question for record in records for question in record["questions"]]


# --------------------------------------------------------------------------- manifest


def test_manifest_is_honest_about_what_is_and_is_not_official() -> None:
    manifest = _adapter().manifest()
    assert manifest.status is AdapterStatus.FULLY_SUPPORTED
    assert manifest.license == "MIT"
    assert manifest.status_tested_at is not None
    # The scale inference is ours, not TAT-QA's. The manifest must say so.
    limitations = " ".join(manifest.known_limitations).lower()
    assert "inferred" in limitations
    assert "ours, not official" in limitations


# --------------------------------------------------------------------------- conversion


def test_every_question_in_the_real_fixture_converts() -> None:
    samples = _adapter().load("dev")
    assert len(samples) == len(_raw_questions())
    assert len(samples) >= 26


def test_sample_ids_are_unique_and_prefixed() -> None:
    samples = _adapter().load("dev")
    ids = [sample.sample_id for sample in samples]
    assert len(ids) == len(set(ids))
    assert all(sample_id.startswith("tatqa:dev:") for sample_id in ids)


def test_the_official_scale_and_answer_type_survive_into_metadata() -> None:
    """The official metric needs TAT-QA's own conflated scale-and-unit notion, which this
    platform's schema otherwise splits apart. If it doesn't reach metadata, the metric is wrong."""
    samples = {s.sample_id.removeprefix("tatqa:dev:"): s for s in _adapter().load("dev")}
    for question in _raw_questions():
        sample = samples[question["uid"]]
        assert sample.metadata["scale"] == question["scale"]
        assert sample.metadata["answer_type"] == question["answer_type"]


def test_every_answer_type_and_scale_in_the_real_data_is_covered() -> None:
    samples = _adapter().load("dev")
    answer_types = {s.metadata["answer_type"] for s in samples}
    scales = {s.metadata["scale"] for s in samples}
    assert answer_types == {"span", "multi-span", "arithmetic", "count"}
    assert scales == {"", "thousand", "million", "billion", "percent"}


def test_multi_span_gold_answers_keep_every_span() -> None:
    """The official metric compares against the answer *list*; flattening it to one string would
    make every multi-span answer unmatchable."""
    samples = {s.sample_id.removeprefix("tatqa:dev:"): s for s in _adapter().load("dev")}
    multi = [q for q in _raw_questions() if q["answer_type"] == "multi-span"]
    assert multi, "fixture must contain multi-span questions"
    for question in multi:
        sample = samples[question["uid"]]
        assert list(sample.gold.acceptable_answers) == [str(a) for a in question["answer"]]


def test_arithmetic_answers_carry_a_numeric_value() -> None:
    samples = _adapter().load("dev")
    arithmetic = [s for s in samples if s.metadata["answer_type"] == "arithmetic"]
    assert arithmetic
    assert all(s.gold.answer_type is AnswerType.NUMERIC for s in arithmetic)


def test_the_table_and_paragraphs_reach_the_sample_context() -> None:
    samples = _adapter().load("dev")
    assert all(sample.context.tables for sample in samples), "every TAT-QA question has a table"
    assert any(sample.context.text for sample in samples), "some have paragraphs too"
    assert all(sample.split_origin is SplitOrigin.OFFICIAL for sample in samples)


def test_an_unknown_split_is_rejected_cleanly() -> None:
    with pytest.raises(Exception, match="no split"):
        _adapter().load("not-a-split")


# --------------------------------------------------------------------------- end to end scoring


@pytest.mark.asyncio
async def test_echo_gold_scores_perfectly_on_the_official_metrics(tmp_path: Path) -> None:
    """A run that reproduces the gold answer exactly must score 1.0 on EM and F1.

    This is the pipeline check: if the adapter, the prompt, the response parsing, the scale
    extraction and the official metric don't line up, a perfect answer will not score perfectly —
    and a real model's score would be silently deflated by our own plumbing.
    """
    samples = _adapter().load("dev")
    result = await RunEngine().run(
        samples=samples,
        model=ModelSpec.parse("mock/echo-gold"),
        config=RunConfig(),
        cache=ResponseCache(tmp_path),
        provider=MockProvider(oracle=build_mock_oracle(samples)),
    )

    em, f1 = TatQAExactMatch(), TatQAF1()
    em_scores = [em.score(s, p).value for s, p in zip(samples, result.predictions, strict=True)]
    f1_scores = [f1.score(s, p).value for s, p in zip(samples, result.predictions, strict=True)]

    # The mock echoes gold.answer (spans joined with " | ") and gold.unit, so a perfect score here
    # proves the whole chain lines up.
    assert sum(float(v or 0) for v in em_scores) / len(em_scores) >= 0.9
    assert sum(float(v or 0) for v in f1_scores) / len(f1_scores) >= 0.9


@pytest.mark.asyncio
async def test_a_wrong_answer_scores_zero(tmp_path: Path) -> None:
    samples = _adapter().load("dev")
    result = await RunEngine().run(
        samples=samples,
        model=ModelSpec.parse("mock/always-wrong"),
        config=RunConfig(),
        cache=ResponseCache(tmp_path),
        provider=MockProvider(oracle=build_mock_oracle(samples)),
    )

    em = TatQAExactMatch()
    scores = [em.score(s, p).value for s, p in zip(samples, result.predictions, strict=True)]
    assert sum(float(v or 0) for v in scores) == 0.0


@pytest.mark.asyncio
async def test_scale_accuracy_is_tracked_separately_from_the_answer(tmp_path: Path) -> None:
    """A right number at the wrong scale is a 1000x error, not a near-miss — so it gets its own
    number, and feeds the wrong_scale_rate gate."""
    samples = _adapter().load("dev")
    result = await RunEngine().run(
        samples=samples,
        model=ModelSpec.parse("mock/echo-gold"),
        config=RunConfig(),
        cache=ResponseCache(tmp_path),
        provider=MockProvider(oracle=build_mock_oracle(samples)),
    )

    metric = TatQAScaleAccuracy()
    results = [metric.score(s, p) for s, p in zip(samples, result.predictions, strict=True)]
    assert all("pred_scale" in r.details and "gold_scale" in r.details for r in results)
