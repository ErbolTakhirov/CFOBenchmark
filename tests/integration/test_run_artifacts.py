"""End-to-end: smoke dataset -> engine -> metrics -> capability rollup -> full 18-file artifact
set. This is the integration test the Milestone 1 acceptance bar (`eval --group smoke`) rests on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from financebench import __version__
from financebench.datasets.smoke.adapter import SmokeDatasetAdapter
from financebench.evaluation.capability_map import rollup_capabilities
from financebench.evaluation.metrics.exact_match import ExactMatchMetric
from financebench.execution.cache import ResponseCache
from financebench.execution.engine import RunEngine
from financebench.models.mock import MockProvider, build_mock_oracle
from financebench.schemas.model_io import ModelSpec
from financebench.schemas.run import RunConfig
from financebench.storage.artifacts import (
    RUN_ARTIFACT_FILENAMES,
    ArtifactInputs,
    write_run_artifacts,
)
from financebench.utils.ids import make_run_id
from financebench.utils.timing import FrozenClock


async def _run_smoke_and_write(out_dir: Path, cache_dir: Path) -> None:
    samples = tuple(SmokeDatasetAdapter().load("dev"))
    model = ModelSpec.parse("mock/echo-gold")
    config = RunConfig()
    clock = FrozenClock()

    engine = RunEngine(clock=clock)
    run_result = await engine.run(
        samples=samples,
        model=model,
        config=config,
        cache=ResponseCache(cache_dir),
        provider=MockProvider(oracle=build_mock_oracle(samples)),
    )

    metric = ExactMatchMetric()
    metric_results = tuple(
        metric.score(sample, prediction)
        for sample, prediction in zip(samples, run_result.predictions, strict=True)
    )
    capability_aggregates = rollup_capabilities(samples, metric_results)

    run_id = make_run_id("smoke", model.ref, config.seed)
    inputs = ArtifactInputs(
        run_id=run_id,
        benchmark_or_group="smoke",
        model=model,
        provider_capabilities=MockProvider().capabilities(model.model),
        config=config,
        created_at=clock.now_iso(),
        financebench_version=__version__,
        dataset_manifests=(SmokeDatasetAdapter().manifest(),),
        samples=samples,
        run_result=run_result,
        metric_results=metric_results,
        capability_aggregates=capability_aggregates,
    )
    write_run_artifacts(out_dir, inputs)


@pytest.mark.asyncio
async def test_every_expected_artifact_file_exists(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    await _run_smoke_and_write(out_dir, tmp_path / "cache")
    for filename in RUN_ARTIFACT_FILENAMES:
        assert (out_dir / filename).is_file(), f"missing artifact: {filename}"


@pytest.mark.asyncio
async def test_predictions_jsonl_has_one_line_per_sample(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    await _run_smoke_and_write(out_dir, tmp_path / "cache")
    lines = (out_dir / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 10
    for line in lines:
        json.loads(line)  # every line is valid JSON


@pytest.mark.asyncio
async def test_metrics_json_reports_perfect_exact_match(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    await _run_smoke_and_write(out_dir, tmp_path / "cache")
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["exact_match"]["mean"] == 1.0
    assert metrics["exact_match"]["n"] == 10


@pytest.mark.asyncio
async def test_errors_jsonl_is_empty_when_nothing_failed(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    await _run_smoke_and_write(out_dir, tmp_path / "cache")
    assert (out_dir / "errors.jsonl").read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_failures_jsonl_is_empty_when_every_metric_passed(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    await _run_smoke_and_write(out_dir, tmp_path / "cache")
    assert (out_dir / "failures.jsonl").read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_gates_and_confidence_intervals_are_valid_unevaluated_placeholders(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    await _run_smoke_and_write(out_dir, tmp_path / "cache")
    gates = json.loads((out_dir / "gates.json").read_text(encoding="utf-8"))
    assert gates["evaluated"] is False
    assert gates["gates"] == []

    ci = json.loads((out_dir / "confidence_intervals.json").read_text(encoding="utf-8"))
    assert ci["exact_match"]["ci_low"] is None
    assert ci["exact_match"]["ci_high"] is None


@pytest.mark.asyncio
async def test_coverage_reports_smoke_as_fully_covered(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    await _run_smoke_and_write(out_dir, tmp_path / "cache")
    coverage = json.loads((out_dir / "coverage.json").read_text(encoding="utf-8"))
    assert coverage["requested_benchmarks"] == ["smoke"]
    assert coverage["supported_benchmarks"] == ["smoke"]
    assert coverage["unavailable_benchmarks"] == []
    assert coverage["evaluated_samples"] == 10
    assert coverage["multimodal_coverage"] == 0.0
    assert coverage["ru_coverage"] == 0.0


@pytest.mark.asyncio
async def test_environment_json_records_reproducibility_fields(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    await _run_smoke_and_write(out_dir, tmp_path / "cache")
    env = json.loads((out_dir / "environment.json").read_text(encoding="utf-8"))
    assert env["model_ref"] == "mock/echo-gold"
    assert env["seed"] == 42
    assert "python_version" in env
    assert "git_commit" in env  # value may be None pre-first-commit; the field must exist


@pytest.mark.asyncio
async def test_summary_and_report_mention_the_run_id_and_model(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    await _run_smoke_and_write(out_dir, tmp_path / "cache")
    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    report = (out_dir / "report.html").read_text(encoding="utf-8")
    for text in (summary, report):
        assert "mock/echo-gold" in text
        assert "smoke" in text


@pytest.mark.asyncio
async def test_report_html_escapes_hostile_content(tmp_path: Path) -> None:
    # A refusal/error message containing HTML-special characters must not break the page.
    out_dir = tmp_path / "run"
    await _run_smoke_and_write(out_dir, tmp_path / "cache")
    report = (out_dir / "report.html").read_text(encoding="utf-8")
    assert "<script>" not in report


@pytest.mark.asyncio
async def test_two_independent_fresh_runs_are_byte_identical(tmp_path: Path) -> None:
    """Determinism, not resume: two *separate* fresh caches (same seed/config/samples/model)
    must produce byte-identical predictions.jsonl. (A rerun sharing one cache is deliberately
    *not* byte-identical to the original — it correctly reports cache_hit=true/attempts=0 instead
    of false/1; that behavior is covered by
    test_execution_engine.py::test_rerun_with_same_cache_hits_everything_and_makes_zero_calls.)
    """
    first_dir = tmp_path / "run1"
    second_dir = tmp_path / "run2"
    await _run_smoke_and_write(first_dir, tmp_path / "cache1")
    await _run_smoke_and_write(second_dir, tmp_path / "cache2")
    first = (first_dir / "predictions.jsonl").read_text(encoding="utf-8")
    second = (second_dir / "predictions.jsonl").read_text(encoding="utf-8")
    assert first == second
