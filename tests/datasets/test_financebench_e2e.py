"""FinanceBench end to end, against the real downloaded 150-question subset.

No committed fixture: FinanceBench is CC BY-NC-4.0 and the 150 rows are not ours to redistribute.
The suite skips — loudly — when the data has not been fetched, rather than passing against a
fixture that should not exist.

    financebench prepare financebench
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from financebench.datasets.financebench.adapter import FinanceBenchAdapter, answer_shape
from financebench.evaluation.grounding import (
    FinanceBenchAnswerAccuracy,
    UnsupportedNumericClaim,
    number_is_supported,
    numbers_in,
)
from financebench.execution.cache import ResponseCache
from financebench.execution.engine import RunEngine
from financebench.models.mock import MockProvider, build_mock_oracle
from financebench.schemas.manifest import AdapterStatus
from financebench.schemas.model_io import ModelSpec
from financebench.schemas.run import RunConfig

DATA = Path("data/downloads/financebench")

pytestmark = pytest.mark.skipif(
    not (DATA / "financebench_open_source.jsonl").is_file(),
    reason="FinanceBench not downloaded (CC BY-NC-4.0, not redistributable). "
    "Run: financebench prepare financebench",
)


def _adapter() -> FinanceBenchAdapter:
    return FinanceBenchAdapter(data_dir=DATA)


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in (DATA / "financebench_open_source.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------- manifest


def test_it_is_labelled_a_public_subset_and_never_the_full_financebench() -> None:
    manifest = _adapter().manifest()
    assert manifest.status is AdapterStatus.SUPPORTED_PUBLIC_SUBSET

    limitations = " ".join(manifest.known_limitations)
    assert "150 of the 10,231" in limitations
    assert "NO EVALUATOR" in limitations, (
        "FinanceBench ships no evaluator — every metric here is ours, and the manifest must say so"
    )


# --------------------------------------------------------------------------- conversion


def test_all_150_convert() -> None:
    samples = _adapter().load("open_source")
    assert len(samples) == 150 == len(_rows())


def test_sample_ids_are_unique_and_prefixed() -> None:
    ids = [s.sample_id for s in _adapter().load("open_source")]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("financebench:open_source:") for i in ids)


def test_the_three_answer_shapes_are_all_present_and_classified() -> None:
    """Conflating them is how a benchmark ends up exact-matching an essay."""
    samples = _adapter().load("open_source")
    shapes = [s.metadata["answer_shape"] for s in samples]

    assert shapes.count("numeric") == 52
    assert shapes.count("boolean") == 37
    assert shapes.count("analytical") == 61
    assert (
        sum([shapes.count("numeric"), shapes.count("boolean"), shapes.count("analytical")]) == 150
    )


def test_gold_evidence_carries_document_and_page_for_retrieval_scoring() -> None:
    samples = _adapter().load("open_source")
    with_pages = [s for s in samples if any(e.page is not None for e in s.gold.evidence)]
    assert len(with_pages) > 140, "retrieval cannot be scored without gold page numbers"
    assert all(e.document_id for s in with_pages for e in s.gold.evidence)


def test_the_justification_is_evaluator_only_and_never_reaches_the_prompt() -> None:
    """FinanceBench's `justification` explains *how to get the answer*. Showing it to the model
    would be handing over the working."""
    from financebench.prompts.renderer import render_messages

    samples = [s for s in _adapter().load("open_source") if s.metadata.get("justification")]
    assert samples

    for sample in samples[:20]:
        prompt = "\n".join(m.content for m in render_messages(sample))
        assert sample.metadata["justification"][:80] not in prompt


# --------------------------------------------------------------------------- the hallucination
# detector — the most important metric here


def test_a_number_stated_from_nowhere_is_caught() -> None:
    evidence = numbers_in("Purchases of property, plant and equipment (1,577)")
    assert number_is_supported(1577.0, evidence)
    assert number_is_supported(-1577.0, evidence)  # parenthesised negative
    assert not number_is_supported(1600.0, evidence), "a plausible invented figure must be caught"


def test_a_correctly_scaled_answer_is_not_a_hallucination() -> None:
    """The filing says 1,577 in a table headed (millions); the model says $1,577 million. Same
    fact. Flagging that would make the metric useless."""
    evidence = numbers_in("Capital expenditures 1,577")
    assert number_is_supported(1_577_000_000.0, evidence)


def test_a_refusal_makes_no_numeric_claim() -> None:
    """A model that declines has invented nothing. Whether it *should* have declined is the
    calibration metric's business."""
    from financebench.execution.engine import build_request
    from financebench.schemas.model_io import FinancialAnswer, ModelResponse
    from financebench.schemas.prediction import Prediction

    sample = _adapter().load("open_source")[0]
    request = build_request(sample, ModelSpec.parse("ollama/qwen2.5:3b"), RunConfig())
    prediction = Prediction(
        sample_id=sample.sample_id,
        benchmark="financebench",
        split="open_source",
        request=request,
        created_at="t",
        response=ModelResponse(
            provider="ollama",
            model="qwen2.5:3b",
            content="{}",
            financial_answer=FinancialAnswer(answer="I cannot tell", insufficient_information=True),
            parsed=True,
        ),
    )
    result = UnsupportedNumericClaim().score(sample, prediction)
    assert result.passed is True


# --------------------------------------------------------------------------- not-applicable is
# not zero


@pytest.mark.asyncio
async def test_an_unscoreable_analytical_answer_is_not_applicable_not_a_failure(
    tmp_path: Path,
) -> None:
    """61 of the 150 gold answers are multi-sentence analyses. Exact-matching an essay is theatre.
    Reporting a fabricated 0.0 for an answer nobody could check is exactly what this project
    exists to refuse — so they return passed=None, and are excluded from the mean rather than
    dragging it down."""
    samples = [
        s for s in _adapter().load("open_source") if s.metadata["answer_shape"] == "analytical"
    ][:5]
    assert samples

    result = await RunEngine().run(
        samples=samples,
        model=ModelSpec.parse("mock/echo-gold"),
        config=RunConfig(),
        cache=ResponseCache(tmp_path),
        provider=MockProvider(oracle=build_mock_oracle(samples)),
    )
    metric = FinanceBenchAnswerAccuracy()
    scored = [metric.score(s, p) for s, p in zip(samples, result.predictions, strict=True)]

    assert all(r.passed is None for r in scored)
    assert all(r.value is None for r in scored)
    assert all("not deterministically checkable" in str(r.details) for r in scored)


@pytest.mark.asyncio
async def test_a_perfect_answer_scores_perfectly_on_the_checkable_ones(tmp_path: Path) -> None:
    """If a gold answer echoed back does not score, the metric is measuring our plumbing."""
    samples = [
        s
        for s in _adapter().load("open_source")
        if s.metadata["answer_shape"] in ("numeric", "boolean")
    ][:30]

    result = await RunEngine().run(
        samples=samples,
        model=ModelSpec.parse("mock/echo-gold"),
        config=RunConfig(),
        cache=ResponseCache(tmp_path),
        provider=MockProvider(oracle=build_mock_oracle(samples)),
    )
    metric = FinanceBenchAnswerAccuracy()
    passed = [metric.score(s, p).passed for s, p in zip(samples, result.predictions, strict=True)]

    assert sum(bool(p) for p in passed) / len(passed) >= 0.9


def test_answer_shape_classification() -> None:
    assert answer_shape("$1577.00") == "numeric"
    assert answer_shape("Yes, the company is capital intensive.") == "boolean"
    assert answer_shape("No.") == "boolean"
    assert (
        answer_shape(
            "Operating margin decreased by 1.7% primarily due to a fall in gross margin, "
            "offset partially by lower SG&A."
        )
        == "analytical"
    )
