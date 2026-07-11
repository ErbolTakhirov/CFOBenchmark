"""Writes every run artifact under ``runs/{run_id}/`` — the full 18-file set the platform's
run-artifact contract specifies.

Milestone 1 populates every file with real data except ``gates.json`` and
``confidence_intervals.json``, which are valid-but-empty placeholders (gate thresholds and
bootstrap confidence intervals are Milestone 6's scoring work) — a real, forward-compatible
instance of each schema, not a bare ``{}`` a later milestone would have to redefine. Nothing here
computes scores; it accepts already-computed predictions/metrics/capability aggregates and only
serializes them, so this module has nothing to get wrong about scoring logic itself.
"""

from __future__ import annotations

import hashlib
import html
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from financebench.evaluation.capability_map import CapabilityDimension
from financebench.evaluation.metrics.base import aggregate_metric
from financebench.execution.engine import RunResult
from financebench.models.base import ProviderCapabilities
from financebench.prompts.renderer import PROMPT_VERSION, SYSTEM_PROMPT
from financebench.schemas.common import RunType
from financebench.schemas.gates import GatesReport
from financebench.schemas.manifest import AdapterStatus, DatasetManifest
from financebench.schemas.metric import MetricAggregate, MetricResult
from financebench.schemas.model_io import ModelSpec
from financebench.schemas.prediction import ParsedAnswer, Prediction
from financebench.schemas.run import RunConfig
from financebench.schemas.sample import CanonicalSample
from financebench.storage.jsonl import write_jsonl, write_model_json, write_model_list_json
from financebench.utils.gitmeta import git_commit, git_is_dirty, os_name, python_version

__all__ = ["RUN_ARTIFACT_FILENAMES", "ArtifactInputs", "write_run_artifacts"]

RUN_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "run_config.json",
    "environment.json",
    "dataset_manifest.json",
    "model_manifest.json",
    "prompt_manifest.json",
    "predictions.jsonl",
    "parsed_answers.jsonl",
    "metric_details.jsonl",
    "errors.jsonl",
    "failures.jsonl",
    "metrics.json",
    "capabilities.json",
    "gates.json",
    "confidence_intervals.json",
    "costs.json",
    "coverage.json",
    "summary.md",
    "report.html",
)


@dataclass(frozen=True)
class ArtifactInputs:
    """Everything needed to write one run's artifacts — the output of `eval`, not a schema
    that's itself persisted (each field lands in its own file, some in more than one shape)."""

    run_id: str
    benchmark_or_group: str
    model: ModelSpec
    provider_capabilities: ProviderCapabilities
    config: RunConfig
    created_at: str
    financebench_version: str
    dataset_manifests: tuple[DatasetManifest, ...]
    samples: tuple[CanonicalSample, ...]
    run_result: RunResult
    metric_results: tuple[MetricResult, ...]
    capability_aggregates: Mapping[CapabilityDimension, MetricAggregate]
    run_type: RunType = RunType.REAL

    @property
    def eligible_for_leaderboard(self) -> bool:
        """A mock run exercised the pipeline. It did not evaluate a model, so it cannot be ranked
        against one."""
        return self.run_type is RunType.REAL


@dataclass(frozen=True)
class _Coverage:
    requested_benchmarks: tuple[str, ...]
    supported_benchmarks: tuple[str, ...]
    unavailable_benchmarks: tuple[str, ...]
    evaluated_samples: int
    skipped_samples: int
    text_only_coverage: float | None
    multimodal_coverage: float | None
    ru_coverage: float | None
    agentic_coverage: float | None


def write_run_artifacts(out_dir: str | Path, inputs: ArtifactInputs) -> None:
    """Write all 18 run artifacts to ``out_dir`` (typically ``runs/{run_id}/``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_model_json(out / "run_config.json", inputs.config)
    _write_environment(out, inputs)
    write_model_list_json(out / "dataset_manifest.json", inputs.dataset_manifests)
    _write_model_manifest(out, inputs)
    _write_prompt_manifest(out)
    write_jsonl(out / "predictions.jsonl", inputs.run_result.predictions)
    write_jsonl(
        out / "parsed_answers.jsonl",
        (_parsed_answer(p) for p in inputs.run_result.predictions),
    )
    write_jsonl(out / "metric_details.jsonl", inputs.metric_results)
    write_jsonl(
        out / "errors.jsonl", (p for p in inputs.run_result.predictions if p.response is None)
    )
    write_jsonl(out / "failures.jsonl", (r for r in inputs.metric_results if r.passed is False))
    metrics_by_name = _write_metrics(out, inputs)
    _write_capabilities(out, inputs)
    write_model_json(out / "gates.json", GatesReport())
    _write_confidence_intervals(out, inputs, metrics_by_name)
    _write_costs(out, inputs)
    coverage = _compute_coverage(inputs)
    _write_coverage(out, coverage)
    _write_summary_md(out, inputs, metrics_by_name, coverage)
    _write_report_html(out, inputs, metrics_by_name, coverage)


def _parsed_answer(prediction: Prediction) -> ParsedAnswer:
    response = prediction.response
    if response is None:
        return ParsedAnswer(sample_id=prediction.sample_id, parse_success=False, raw_text="")
    return ParsedAnswer(
        sample_id=prediction.sample_id,
        financial_answer=response.financial_answer,
        parse_success=response.parsed,
        raw_text=response.content,
    )


def _write_environment(out: Path, inputs: ArtifactInputs) -> None:
    payload = {
        "run_id": inputs.run_id,
        "financebench_version": inputs.financebench_version,
        "created_at": inputs.created_at,
        "benchmark_or_group": inputs.benchmark_or_group,
        "model_ref": inputs.model.ref,
        "provider": inputs.model.provider,
        # The leaderboard builder reads these two fields and nothing else to decide whether a run
        # may be ranked — so a mock run is excluded by data, not by a name-matching heuristic.
        "run_type": inputs.run_type.value,
        "eligible_for_leaderboard": inputs.eligible_for_leaderboard,
        "seed": inputs.config.seed,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "python_version": python_version(),
        "os": os_name(),
    }
    (out / "environment.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_model_manifest(out: Path, inputs: ArtifactInputs) -> None:
    payload = {
        "model": inputs.model.model_dump(mode="json"),
        "capabilities": asdict(inputs.provider_capabilities),
    }
    (out / "model_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_prompt_manifest(out: Path) -> None:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    }
    (out / "prompt_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _write_metrics(out: Path, inputs: ArtifactInputs) -> dict[str, MetricAggregate]:
    by_name: dict[str, list[MetricResult]] = defaultdict(list)
    for result in inputs.metric_results:
        by_name[result.metric_name].append(result)
    aggregates = {name: aggregate_metric(name, results) for name, results in by_name.items()}
    payload = {name: agg.model_dump(mode="json") for name, agg in aggregates.items()}
    (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return aggregates


def _write_capabilities(out: Path, inputs: ArtifactInputs) -> None:
    payload = {
        dimension.value: aggregate.model_dump(mode="json")
        for dimension, aggregate in inputs.capability_aggregates.items()
    }
    (out / "capabilities.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_confidence_intervals(
    out: Path, inputs: ArtifactInputs, metrics_by_name: Mapping[str, MetricAggregate]
) -> None:
    payload = {name: agg.model_dump(mode="json") for name, agg in metrics_by_name.items()}
    payload.update(
        {
            f"capability:{dimension.value}": agg.model_dump(mode="json")
            for dimension, agg in inputs.capability_aggregates.items()
        }
    )
    (out / "confidence_intervals.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _write_costs(out: Path, inputs: ArtifactInputs) -> None:
    result = inputs.run_result
    cache_hit_rate = (result.n_cache_hits / result.n_samples) if result.n_samples else None
    payload = {
        "total_estimated_cost_usd": result.total_estimated_cost_usd,
        "total_tokens": result.total_tokens,
        "n_samples": result.n_samples,
        "n_errors": result.n_errors,
        "n_cache_hits": result.n_cache_hits,
        "cache_hit_rate": cache_hit_rate,
        "budget_exceeded": result.budget_exceeded,
    }
    (out / "costs.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _compute_coverage(inputs: ArtifactInputs) -> _Coverage:
    requested = tuple(sorted({manifest.name for manifest in inputs.dataset_manifests}))
    unavailable = tuple(
        sorted(
            manifest.name
            for manifest in inputs.dataset_manifests
            if manifest.status is AdapterStatus.UNAVAILABLE
        )
    )
    supported = tuple(sorted({sample.benchmark for sample in inputs.samples}))
    total = len(inputs.samples)
    n_multimodal = sum(1 for sample in inputs.samples if sample.context.images)
    n_ru = sum(1 for sample in inputs.samples if sample.language != "en")
    n_agentic = sum(1 for sample in inputs.samples if sample.tools)
    return _Coverage(
        requested_benchmarks=requested,
        supported_benchmarks=supported,
        unavailable_benchmarks=unavailable,
        evaluated_samples=total,
        skipped_samples=0,
        text_only_coverage=((total - n_multimodal) / total) if total else None,
        multimodal_coverage=(n_multimodal / total) if total else None,
        ru_coverage=(n_ru / total) if total else None,
        agentic_coverage=(n_agentic / total) if total else None,
    )


def _write_coverage(out: Path, coverage: _Coverage) -> None:
    (out / "coverage.json").write_text(
        json.dumps(asdict(coverage), indent=2) + "\n", encoding="utf-8"
    )


def _format_mean(aggregate: MetricAggregate) -> str:
    return f"{aggregate.mean:.3f}" if aggregate.mean is not None else "n/a"


#: Shown at the very top of every mock report, before any number. A reader who skims one line of
#: this document must not come away thinking a model was measured.
MOCK_WATERMARK_TITLE = "MOCK — NOT A MODEL RESULT"
MOCK_WATERMARK_BODY = (
    "This run used the `mock` provider, a simulator that is handed the gold answers. Its scores "
    "measure whether the pipeline works — they say nothing whatsoever about any model's financial "
    "ability. This run is barred from the leaderboard and from the Finance Capability Index."
)


def _markdown_watermark(inputs: ArtifactInputs) -> list[str]:
    if inputs.run_type is RunType.REAL:
        return []
    return [f"> ## ⚠️ {MOCK_WATERMARK_TITLE}", ">", f"> {MOCK_WATERMARK_BODY}", ""]


def _html_watermark(inputs: ArtifactInputs) -> str:
    if inputs.run_type is RunType.REAL:
        return ""
    return (
        f'<div class="mock-watermark"><h2>⚠️ {html.escape(MOCK_WATERMARK_TITLE)}</h2>'
        f"<p>{html.escape(MOCK_WATERMARK_BODY)}</p></div>"
    )


def _write_summary_md(
    out: Path,
    inputs: ArtifactInputs,
    metrics_by_name: Mapping[str, MetricAggregate],
    coverage: _Coverage,
) -> None:
    result = inputs.run_result
    lines = [
        f"# FinanceBecnh run summary — `{inputs.run_id}`",
        "",
        *_markdown_watermark(inputs),
        f"- **Model:** `{inputs.model.ref}`",
        f"- **Run type:** `{inputs.run_type.value}`"
        f"{'' if inputs.eligible_for_leaderboard else ' (not leaderboard-eligible)'}",
        f"- **Benchmark/group:** `{inputs.benchmark_or_group}`",
        f"- **Samples evaluated:** {result.n_samples} "
        f"(errors: {result.n_errors}, cache hits: {result.n_cache_hits})",
        f"- **Created at:** {inputs.created_at}",
        f"- **FinanceBecnh version:** {inputs.financebench_version}",
        "",
        "## Metrics",
        "",
        "| Metric | n | Mean |",
        "|---|---|---|",
    ]
    for name, agg in sorted(metrics_by_name.items()):
        lines.append(f"| {name} | {agg.n} | {_format_mean(agg)} |")
    lines += ["", "## Capability dimensions", "", "| Dimension | n | Mean |", "|---|---|---|"]
    for dimension, agg in sorted(inputs.capability_aggregates.items(), key=lambda kv: kv[0].value):
        lines.append(f"| {dimension.value} | {agg.n} | {_format_mean(agg)} |")
    lines += [
        "",
        "## Coverage",
        "",
        f"- Requested benchmarks: {', '.join(coverage.requested_benchmarks) or 'none'}",
        f"- Supported benchmarks: {', '.join(coverage.supported_benchmarks) or 'none'}",
        f"- Unavailable benchmarks: {', '.join(coverage.unavailable_benchmarks) or 'none'}",
        "",
        "## Reproduce this run",
        "",
        "```bash",
        f"python -m financebench.cli eval --group {inputs.benchmark_or_group} "
        f"--model-config <config used> --seed {inputs.config.seed}",
        "```",
        "",
        "_The Finance Capability Index, critical gates, and confidence intervals are not yet "
        "computed (Milestone 6) — `gates.json` and `confidence_intervals.json` are valid but "
        "empty placeholders in this release._",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_report_html(
    out: Path,
    inputs: ArtifactInputs,
    metrics_by_name: Mapping[str, MetricAggregate],
    coverage: _Coverage,
) -> None:
    result = inputs.run_result

    def esc(value: object) -> str:
        return html.escape(str(value))

    metric_rows = (
        "\n".join(
            f"<tr><td>{esc(name)}</td><td>{agg.n}</td><td>{_format_mean(agg)}</td></tr>"
            for name, agg in sorted(metrics_by_name.items())
        )
        or "<tr><td colspan='3'>No metrics computed.</td></tr>"
    )
    capability_rows = (
        "\n".join(
            f"<tr><td>{esc(dimension.value)}</td><td>{agg.n}</td><td>{_format_mean(agg)}</td></tr>"
            for dimension, agg in sorted(
                inputs.capability_aggregates.items(), key=lambda kv: kv[0].value
            )
        )
        or "<tr><td colspan='3'>No capability dimensions mapped.</td></tr>"
    )
    error_predictions = [p for p in result.predictions if p.response is None][:20]
    error_rows = (
        "\n".join(
            f"<tr><td>{esc(p.sample_id)}</td><td>{esc(p.error_type)}</td><td>{esc(p.error)}</td></tr>"
            for p in error_predictions
        )
        or "<tr><td colspan='3'>No errors.</td></tr>"
    )

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FinanceBecnh report — {esc(inputs.run_id)}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
          margin: 2rem auto; max-width: 900px; color: #1a1a1a; line-height: 1.5; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; }}
  th {{ background: #f4f4f4; }}
  code {{ background: #f4f4f4; padding: 0.1rem 0.3rem; border-radius: 3px; }}
  .badge {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px;
            background: #eef2ff; color: #3730a3; font-size: 0.85rem; }}
  .note {{ color: #555; font-size: 0.9rem; }}
  .mock-watermark {{ border: 3px solid #b91c1c; background: #fef2f2; color: #7f1d1d;
                     padding: 1rem 1.25rem; margin: 1.5rem 0; border-radius: 6px; }}
  .mock-watermark h2 {{ margin-top: 0; }}
</style>
</head>
<body>
  <h1>FinanceBecnh run report</h1>
  {_html_watermark(inputs)}
  <p><span class="badge">{esc(inputs.run_id)}</span></p>
  <ul>
    <li><strong>Model:</strong> <code>{esc(inputs.model.ref)}</code></li>
    <li><strong>Run type:</strong> <code>{esc(inputs.run_type.value)}</code></li>
    <li><strong>Benchmark/group:</strong> <code>{esc(inputs.benchmark_or_group)}</code></li>
    <li><strong>Samples evaluated:</strong> {result.n_samples}
        (errors: {result.n_errors}, cache hits: {result.n_cache_hits})</li>
    <li><strong>Created at:</strong> {esc(inputs.created_at)}</li>
  </ul>

  <h2>Metrics</h2>
  <table>
    <tr><th>Metric</th><th>n</th><th>Mean</th></tr>
    {metric_rows}
  </table>

  <h2>Capability dimensions</h2>
  <p class="note">The Finance Capability Index, critical gates, and confidence intervals are
     not yet computed (Milestone 6) — this section shows raw per-dimension means only.</p>
  <table>
    <tr><th>Dimension</th><th>n</th><th>Mean</th></tr>
    {capability_rows}
  </table>

  <h2>Errors</h2>
  <table>
    <tr><th>Sample</th><th>Type</th><th>Message</th></tr>
    {error_rows}
  </table>

  <h2>Coverage</h2>
  <ul>
    <li>Requested benchmarks: {esc(", ".join(coverage.requested_benchmarks) or "none")}</li>
    <li>Supported benchmarks: {esc(", ".join(coverage.supported_benchmarks) or "none")}</li>
    <li>Unavailable benchmarks: {esc(", ".join(coverage.unavailable_benchmarks) or "none")}</li>
  </ul>
</body>
</html>
"""
    (out / "report.html").write_text(doc, encoding="utf-8")
