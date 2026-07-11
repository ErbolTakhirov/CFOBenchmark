"""FinanceReasoning end to end, against the **real downloaded data**.

There is no committed fixture here, and that is not an oversight: the upstream repository ships no
LICENCE file and states no licence in its README, so the default is "all rights reserved" and the
data cannot be redistributed. The adapter therefore downloads at runtime, and this suite **skips —
loudly — when the data has not been fetched**, rather than quietly passing against a fixture that
should not exist.

    financebench prepare finance_reasoning
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from financebench.datasets.finance_reasoning.adapter import FinanceReasoningAdapter
from financebench.evaluation.native.finance_reasoning import FinanceReasoningAccuracy
from financebench.execution.cache import ResponseCache
from financebench.execution.engine import RunEngine
from financebench.models.mock import MockProvider, build_mock_oracle
from financebench.schemas.common import AnswerType
from financebench.schemas.manifest import AdapterStatus
from financebench.schemas.model_io import ModelSpec
from financebench.schemas.run import RunConfig

DATA_DIR = Path("data/downloads/finance_reasoning")

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "easy.json").is_file(),
    reason=(
        "FinanceReasoning data not downloaded. It is NOT bundled — upstream ships no licence, so "
        "it cannot be redistributed. Run: financebench prepare finance_reasoning"
    ),
)

LEVELS = ("easy", "medium", "hard")
#: The real, official counts — asserted, so a silently-truncated download is caught.
EXPECTED_COUNTS = {"easy": 1000, "medium": 1000, "hard": 238}


def _adapter() -> FinanceReasoningAdapter:
    return FinanceReasoningAdapter(data_dir=DATA_DIR)


def test_manifest_records_the_licence_blocker_rather_than_glossing_over_it() -> None:
    manifest = _adapter().manifest()
    assert manifest.status is AdapterStatus.FULLY_SUPPORTED
    assert manifest.redistribution_status == "not_redistributable"
    limitations = " ".join(manifest.known_limitations).lower()
    assert "licence blocker" in limitations or "license blocker" in limitations
    assert "eval()" in " ".join(manifest.known_limitations)


@pytest.mark.parametrize("level", LEVELS)
def test_every_record_of_the_real_dataset_converts(level: str) -> None:
    samples = _adapter().load(level)
    raw = json.loads((DATA_DIR / f"{level}.json").read_text(encoding="utf-8"))

    assert len(samples) == len(raw), "a dropped record is a silent coverage loss"
    assert len(samples) == EXPECTED_COUNTS[level]


@pytest.mark.parametrize("level", LEVELS)
def test_sample_ids_are_unique_and_prefixed(level: str) -> None:
    ids = [sample.sample_id for sample in _adapter().load(level)]
    assert len(ids) == len(set(ids))
    assert all(sample_id.startswith(f"finance_reasoning:{level}:") for sample_id in ids)


def test_the_official_ground_truth_and_tolerance_travel_with_the_sample() -> None:
    samples = _adapter().load("easy")
    for sample in samples[:50]:
        assert sample.metadata["ground_truth"]
        # The official rule: eps = |gt| * 0.002.
        assert sample.evaluation.relative_tolerance == 0.002
        assert sample.gold.answer_type in (AnswerType.NUMERIC, AnswerType.BOOLEAN)


def test_both_context_shapes_in_the_real_data_are_handled() -> None:
    """FinanceReasoning mixes two context shapes, and both must survive conversion.

    Some records carry a JSON-encoded dataframe (the gold python_solution indexes it as
    df["GBP"]["FY 2019"]) — those are rendered as a real table, because a model shown raw JSON has
    to parse it before it can reason. Others carry prose ("notes to consolidated financial
    statements ..."). Neither may end up with an empty context: a sample with no context is
    unanswerable, and would silently drag the score down for reasons that have nothing to do with
    the model.
    """
    samples = _adapter().load("easy")
    with_tables = [s for s in samples if s.context.tables]
    with_prose = [s for s in samples if s.context.text and not s.context.tables]

    assert with_tables, "the dataframe-shaped contexts must render as tables"
    assert with_prose, "the prose contexts must survive too"


def test_upstream_records_with_no_context_are_kept_but_flagged() -> None:
    """6 of the 1,000 'easy' records ship an empty context upstream. The question is unanswerable
    — there is nothing to answer it from.

    They are KEPT, because dropping them would break count-parity with published FinanceReasoning
    numbers (which include them). But they are flagged, so a reader can see that a ~0.6% floor is
    baked into the benchmark and is not the model's failing.
    """
    samples = _adapter().load("easy")
    empty = [s for s in samples if not s.context.tables and not s.context.text]

    assert empty, "the upstream data really does contain context-less records"
    assert len(empty) < len(samples) * 0.02, "but only a handful"
    assert all(s.metadata["context_empty"] == "true" for s in empty)
    assert all(
        s.metadata["context_empty"] == "false"
        for s in samples
        if s.context.tables or s.context.text
    )


def test_the_gold_python_solution_is_kept_for_grading_but_never_shown_to_the_model() -> None:
    """It is on `gold.program`, which the prompt renderer never reads — asserted structurally in
    tests/security/test_gold_answer_leakage.py."""
    samples = _adapter().load("easy")
    with_program = [s for s in samples if s.gold.program]
    assert with_program, "records carry a gold python_solution"

    from financebench.prompts.renderer import render_messages

    sample = with_program[0]
    rendered = "\n".join(m.content for m in render_messages(sample))
    assert sample.gold.program not in rendered


@pytest.mark.asyncio
async def test_echo_gold_scores_perfectly_and_a_wrong_answer_scores_zero(tmp_path: Path) -> None:
    """A perfect answer must score 1.0 — otherwise the metric is measuring our plumbing, not the
    model."""
    samples = list(_adapter().load("hard"))[:40]
    metric = FinanceReasoningAccuracy()

    perfect = await RunEngine().run(
        samples=samples,
        model=ModelSpec.parse("mock/echo-gold"),
        config=RunConfig(),
        cache=ResponseCache(tmp_path / "a"),
        provider=MockProvider(oracle=build_mock_oracle(samples)),
    )
    scores = [metric.score(s, p).passed for s, p in zip(samples, perfect.predictions, strict=True)]
    assert all(scores), "echo-gold must score perfectly on the official 0.2% tolerance"

    wrong = await RunEngine().run(
        samples=samples,
        model=ModelSpec.parse("mock/always-wrong"),
        config=RunConfig(),
        cache=ResponseCache(tmp_path / "b"),
        provider=MockProvider(oracle=build_mock_oracle(samples)),
    )
    wrong_scores = [
        metric.score(s, p).passed for s, p in zip(samples, wrong.predictions, strict=True)
    ]
    assert not any(wrong_scores)


@pytest.mark.asyncio
async def test_the_02_percent_tolerance_is_actually_applied(tmp_path: Path) -> None:
    """The whole point of the native metric: a 0.1% error passes, a 1% error does not."""
    from financebench.evaluation.native.finance_reasoning import get_acc

    assert get_acc(100.15, 100.0) == 1, "0.15% off is within the 0.2% tolerance"
    assert get_acc(100.25, 100.0) == 0, "0.25% off is outside it"
    # Relative to the gold, so a gold of exactly zero admits only an exact zero.
    assert get_acc(0.0, 0.0) == 1
    assert get_acc(0.0001, 0.0) == 0
