"""End-to-end proof that the FinQA adapter's `fully_supported` claim is real: it loads real
(not synthetic) data, the program executor reproduces every gold execution result, and running
the fixture through the actual engine + native metric produces correct, gradable scores.

Uses `tests/fixtures/finqa/` — a small, legally-redistributable slice of the real official test
split (see that directory's README for provenance) — not the full dataset, which
`financebench prepare finqa` downloads separately for real evaluation runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from financebench.datasets.finqa.adapter import FinQAAdapter
from financebench.evaluation.native.finqa import (
    FinQAExecutionAccuracy,
    execute_program,
    str_to_num_finqa,
)
from financebench.execution.cache import ResponseCache
from financebench.execution.engine import RunEngine
from financebench.models.mock import MockProvider, build_mock_oracle
from financebench.schemas.manifest import AdapterStatus
from financebench.schemas.model_io import ModelSpec
from financebench.schemas.run import RunConfig
from financebench.utils.errors import DatasetLoadError

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "finqa"


def _adapter() -> FinQAAdapter:
    return FinQAAdapter(data_dir=_FIXTURE_DIR)


def _raw_records() -> list[dict[str, object]]:
    raw: list[dict[str, object]] = json.loads(
        (_FIXTURE_DIR / "test.json").read_text(encoding="utf-8")
    )
    return raw


# --------------------------------------------------------------------------- manifest


def test_manifest_declares_fully_supported_and_is_tested() -> None:
    manifest = _adapter().manifest()
    assert manifest.status is AdapterStatus.FULLY_SUPPORTED
    assert manifest.status_tested_at is not None
    assert "test" in manifest.local_splits


# --------------------------------------------------------------------------- loading


def test_load_returns_every_fixture_sample() -> None:
    samples = _adapter().load("test")
    assert len(samples) == len(_raw_records())
    assert {s.benchmark for s in samples} == {"finqa"}
    assert {s.split for s in samples} == {"test"}


def test_sample_ids_are_unique_and_correctly_prefixed() -> None:
    samples = _adapter().load("test")
    ids = [s.sample_id for s in samples]
    assert len(set(ids)) == len(ids)
    assert all(sid.startswith("finqa:test:") for sid in ids)


def test_gold_numeric_value_matches_official_exe_ans() -> None:
    samples = {s.sample_id.removeprefix("finqa:test:"): s for s in _adapter().load("test")}
    for record in _raw_records():
        from financebench.utils.ids import slugify

        sample = samples[slugify(str(record["id"]))]
        exe_ans = record["qa"]["exe_ans"]
        if isinstance(exe_ans, str):
            # `greater`-op records: exe_ans is "yes"/"no", not a number — the adapter correctly
            # leaves numeric_value unset for these (see test_finqa_e2e's boolean-mode coverage).
            assert sample.gold.numeric_value is None
        else:
            assert sample.gold.numeric_value == pytest.approx(exe_ans)
        assert sample.gold.program == record["qa"]["program"]


def test_context_includes_the_table_and_surrounding_text() -> None:
    samples = _adapter().load("test")
    with_tables = [s for s in samples if s.context.tables]
    assert with_tables, "expected at least one sample with a table in context"
    sample = with_tables[0]
    assert len(sample.context.tables[0].rows) > 1
    assert len(sample.context.text) > 0


def test_unknown_split_raises_dataset_load_error() -> None:
    with pytest.raises(DatasetLoadError, match="no split"):
        _adapter().load("bogus_split")


def test_missing_data_dir_raises_a_helpful_error(tmp_path: Path) -> None:
    adapter = FinQAAdapter(data_dir=tmp_path / "does-not-exist")
    with pytest.raises(DatasetLoadError, match="prepare finqa"):
        adapter.load("test")


# --------------------------------------------------------------------------- program executor
# The strongest evidence for `fully_supported`: re-executing every real gold program reproduces
# its recorded official execution result, across every FinQA operation type and both single- and
# multi-step (#N-chained) programs.


def test_str_to_num_handles_percent_and_const_encodings() -> None:
    assert str_to_num_finqa("1,234") == 1234.0
    assert str_to_num_finqa("12%") == pytest.approx(0.12)
    assert str_to_num_finqa("const_100") == 100.0
    assert str_to_num_finqa("const_m1") == -1.0
    assert str_to_num_finqa("not a number") is None


@pytest.mark.parametrize("record", _raw_records(), ids=lambda r: r["id"])
def test_execute_program_reproduces_every_gold_exe_ans(record: dict[str, object]) -> None:
    qa = record["qa"]
    result = execute_program(qa["program"], record["table"])
    assert result is not None, f"executor returned None for {record['id']}: {qa['program']}"
    if isinstance(result, str):
        assert result == qa["exe_ans"]
    else:
        assert result == pytest.approx(qa["exe_ans"], abs=1e-3)


def test_execute_program_covers_every_finqa_operation_in_the_fixture() -> None:
    ops_seen = {record["qa"]["program"].split("(")[0] for record in _raw_records()}
    expected = {
        "add",
        "subtract",
        "multiply",
        "divide",
        "greater",
        "table_max",
        "table_min",
        "table_sum",
        "table_average",
    }
    assert expected <= ops_seen, f"fixture is missing coverage for: {expected - ops_seen}"


def test_execute_program_handles_multi_step_chained_programs() -> None:
    chained = [r for r in _raw_records() if r["qa"]["program"].count("(") > 1]
    assert chained, "expected at least one multi-step program in the fixture"
    for record in chained:
        result = execute_program(record["qa"]["program"], record["table"])
        assert result == pytest.approx(record["qa"]["exe_ans"], abs=1e-3)


def test_execute_program_returns_none_for_malformed_input() -> None:
    assert execute_program("not a program at all") is None
    assert execute_program("unknown_op(1, 2)") is None
    assert execute_program("divide(1, 0)") is None


def test_execute_program_returns_none_for_unknown_table_row() -> None:
    table = (("year", "value"), ("2024", "100"))
    assert execute_program("table_sum(nonexistent row, none)", table) is None


# --------------------------------------------------------------------------- end-to-end scoring


@pytest.mark.asyncio
async def test_echo_gold_scores_perfectly_with_the_native_metric(tmp_path: Path) -> None:
    samples = _adapter().load("test")
    result = await RunEngine().run(
        samples=samples,
        model=ModelSpec.parse("mock/echo-gold"),
        config=RunConfig(),
        cache=ResponseCache(tmp_path),
        provider=MockProvider(oracle=build_mock_oracle(samples)),
    )
    assert result.n_errors == 0

    metric = FinQAExecutionAccuracy()
    scores = [
        metric.score(sample, prediction)
        for sample, prediction in zip(samples, result.predictions, strict=True)
    ]
    assert all(s.passed for s in scores)


@pytest.mark.asyncio
async def test_always_wrong_scores_zero_with_the_native_metric(tmp_path: Path) -> None:
    samples = _adapter().load("test")
    result = await RunEngine().run(
        samples=samples,
        model=ModelSpec.parse("mock/always-wrong"),
        config=RunConfig(),
        cache=ResponseCache(tmp_path),
        provider=MockProvider(oracle=build_mock_oracle(samples)),
    )
    metric = FinQAExecutionAccuracy()
    scores = [
        metric.score(sample, prediction)
        for sample, prediction in zip(samples, result.predictions, strict=True)
    ]
    assert not any(s.passed for s in scores)


@pytest.mark.asyncio
async def test_refuse_profile_is_scored_as_no_extractable_number(tmp_path: Path) -> None:
    samples = _adapter().load("test")
    result = await RunEngine().run(
        samples=samples,
        model=ModelSpec.parse("mock/refuse"),
        config=RunConfig(),
        cache=ResponseCache(tmp_path),
        provider=MockProvider(oracle=build_mock_oracle(samples)),
    )
    metric = FinQAExecutionAccuracy()
    scores = [
        metric.score(sample, prediction)
        for sample, prediction in zip(samples, result.predictions, strict=True)
    ]
    assert all(not s.passed for s in scores)
    # Numeric-type samples fail with an explicit "no extractable number" reason; the one
    # boolean-type (`greater`) sample instead fails the yes/no string comparison — both are
    # correctly `passed=False`, just via different branches of the metric.
    numeric_scores = [s for s in scores if s.details.get("mode") != "boolean"]
    assert numeric_scores
    assert all(
        s.details.get("reason") == "predicted answer has no extractable number"
        for s in numeric_scores
    )
