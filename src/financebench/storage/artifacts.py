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
from dataclasses import asdict, dataclass, field
from pathlib import Path

from financebench.evaluation.capability_map import CapabilityDimension
from financebench.evaluation.failures import FailureRecord, failure_distribution
from financebench.evaluation.fingerprint import current_fingerprint
from financebench.evaluation.metrics.base import aggregate_metric
from financebench.evaluation.scoring import FinanceScores
from financebench.evaluation.stats import bootstrap_ci
from financebench.execution.engine import RunResult
from financebench.models.base import ProviderCapabilities
from financebench.prompts.profiles import create_prompt_profile
from financebench.schemas.common import RunType
from financebench.schemas.gates import GateResult, GatesReport
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

#: Written only by a tool_assisted run. The DURABLE record of what the model asked the tools for and
#: what they said — the only place a "called the calculator and then ignored it" failure is visible.
TOOL_TRACE_FILENAME = "tool_traces.jsonl"


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
    failures: tuple[FailureRecord, ...] = ()
    gates: GatesReport = field(default_factory=GatesReport)
    scores: FinanceScores | None = None
    verdict: str = "NOT_EVALUATED"
    verdict_reasons: tuple[str, ...] = ()
    retrieval: Mapping[str, object] | None = None
    conversation: Mapping[str, object] | None = None

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
    _write_prompt_manifest(out, inputs)
    write_jsonl(out / "predictions.jsonl", inputs.run_result.predictions)
    write_jsonl(
        out / "parsed_answers.jsonl",
        (_parsed_answer(p) for p in inputs.run_result.predictions),
    )
    write_jsonl(out / "metric_details.jsonl", inputs.metric_results)
    write_jsonl(
        out / "errors.jsonl", (p for p in inputs.run_result.predictions if p.response is None)
    )
    write_jsonl(out / "failures.jsonl", inputs.failures)
    metrics_by_name = _write_metrics(out, inputs)
    _write_capabilities(out, inputs)
    write_model_json(out / "gates.json", inputs.gates)
    _write_confidence_intervals(out, inputs, metrics_by_name)
    _write_costs(out, inputs)
    coverage = _compute_coverage(inputs)
    _write_coverage(out, coverage)
    _write_summary_md(out, inputs, metrics_by_name, coverage)
    _write_report_html(out, inputs, metrics_by_name, coverage)

    # A tool_assisted run also writes its traces. Accuracy can say the answer was wrong; only the
    # trace can say the model called the calculator, got the right number, and then wrote a different
    # one — a failure every end-to-end metric misattributes to arithmetic.
    traces = inputs.run_result.tool_traces
    if traces:
        (out / TOOL_TRACE_FILENAME).write_text(
            "\n".join(
                json.dumps(t.to_json() if hasattr(t, "to_json") else t) for t in traces.values()
            )
            + "\n",
            encoding="utf-8",
        )


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
        # The evaluator fingerprint: what OUR code was, not what the model was. Two runs with
        # different digests are not comparable — fixing the answer parser once moved a score from
        # 5% to 15% on identical cached responses, and nothing about the model had changed.
        "evaluator_fingerprint": current_fingerprint().to_json(),
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


def _write_prompt_manifest(out: Path, inputs: ArtifactInputs) -> None:
    """Record exactly what the model was asked for.

    The system prompt is hashed rather than embedded because it varies per sample for some
    profiles (``tool_agent_v1`` names the available tools). Hashing the *first* sample's system
    prompt is enough to detect an edited profile, which is what the hash is for — a profile whose
    text changed without its version changing would silently break comparability.
    """
    profile = create_prompt_profile(inputs.config.prompt_profile)
    system_text = (
        profile.system(inputs.samples[0], inputs.config.eval_mode) if inputs.samples else ""
    )
    payload = {
        "prompt_profile": inputs.config.prompt_profile,
        "eval_mode": inputs.config.eval_mode.value,
        "response_format": profile.response_format,
        "elicits_program": profile.elicits_program,
        "system_prompt_sha256": hashlib.sha256(system_text.encode("utf-8")).hexdigest(),
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
    payload: dict[str, object] = {
        "dimensions": {
            dimension.value: aggregate.model_dump(mode="json")
            for dimension, aggregate in inputs.capability_aggregates.items()
        },
        "scores": inputs.scores.to_json() if inputs.scores else None,
        "verdict": inputs.verdict,
        "verdict_reasons": list(inputs.verdict_reasons),
        "failure_distribution": failure_distribution(inputs.failures),
        "retrieval": dict(inputs.retrieval) if inputs.retrieval else None,
        "conversation": dict(inputs.conversation) if inputs.conversation else None,
    }
    (out / "capabilities.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_confidence_intervals(
    out: Path, inputs: ArtifactInputs, metrics_by_name: Mapping[str, MetricAggregate]
) -> None:
    """Bootstrap 95% confidence intervals for every metric.

    A score without an interval invites a ranking that the evidence does not support. 45% vs 50%
    on 40 samples is noise, and the interval is what says so.
    """
    by_metric: dict[str, list[float]] = defaultdict(list)
    for result in inputs.metric_results:
        if isinstance(result.value, bool):
            by_metric[result.metric_name].append(1.0 if result.value else 0.0)
        elif isinstance(result.value, int | float):
            by_metric[result.metric_name].append(float(result.value))

    payload: dict[str, object] = {}
    for name, values in sorted(by_metric.items()):
        ci = bootstrap_ci(values)
        if ci is None:
            continue
        payload[name] = {
            "mean": round(ci.mean, 4),
            "ci_low": round(ci.ci_low, 4),
            "ci_high": round(ci.ci_high, 4),
            "n": ci.n,
            "underpowered": ci.underpowered,
            "method": "percentile bootstrap, 2000 iterations, seed 42",
        }
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


def _scores_lines(inputs: ArtifactInputs) -> list[str]:
    scores = inputs.scores
    if scores is None:
        return []
    lines = ["### Top-level scores", ""]
    for label, value in (
        ("Financial Core Score (context_given)", scores.core_score),
        ("Financial RAG Score (retrieval_required)", scores.rag_score),
        ("Financial Agent Score (tool_assisted)", scores.agent_score),
    ):
        lines.append(f"- {label}: {value:.3f}" if value is not None else f"- {label}: not measured")
    if scores.fci is not None:
        lines.append(f"- **Finance Capability Index: {scores.fci:.3f}**")
    else:
        lines.append(f"- Finance Capability Index: **withheld** — {scores.fci_withheld_because}")
    lines.append(f"- Reliability penalty applied: x{scores.reliability_penalty:.3f}")
    lines.append("")
    return lines


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
        "## Verdict",
        "",
        f"### `{inputs.verdict}`",
        "",
        *[f"- {reason}" for reason in inputs.verdict_reasons],
        "",
        *_scores_lines(inputs),
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
    # -- critical gates
    if inputs.gates.evaluated:
        lines += [
            "",
            "## Critical gates",
            "",
            "| Gate | Threshold | Observed | Result |",
            "|---|---|---|---|",
        ]
        for gate in inputs.gates.gates:
            mark = "PASS" if gate.passed else "**FAIL**"
            lines.append(f"| {gate.gate_name} | {gate.threshold} | {gate.observed} | {mark} |")
        if inputs.gates.any_critical_gate_failed:
            lines += [
                "",
                "> A **critical** gate failed. However good the average above looks, this model "
                "makes the kind of error that is not a near-miss in a financial context.",
            ]

    # -- failure distribution: *how* it fails, which the mean cannot tell you
    distribution = failure_distribution(inputs.failures)
    if distribution:
        lines += ["", "## How it failed", "", "| Failure | Count |", "|---|---|"]
        for name, count in distribution.items():
            lines.append(f"| {name} | {count} |")

    # -- worst examples, with the model's own words
    worst = [f for f in inputs.failures if f.catastrophic][:5] or list(inputs.failures)[:5]
    if worst:
        lines += ["", "## Representative failures", ""]
        for failure in worst:
            lines += [
                f"- **{failure.sample_id}** ({failure.failure_type.value})",
                f"  - Q: {failure.question[:160]}",
                f"  - gold: `{failure.gold[:80]}` · model: `{failure.predicted[:80]}`",
            ]

    lines += [
        "",
        "## Coverage",
        "",
        f"- Requested benchmarks: {', '.join(coverage.requested_benchmarks) or 'none'}",
        f"- Supported benchmarks: {', '.join(coverage.supported_benchmarks) or 'none'}",
        f"- Unavailable benchmarks: {', '.join(coverage.unavailable_benchmarks) or 'none'}",
        f"- Samples scored: {coverage.evaluated_samples}",
        "",
        "## Reproduce this run",
        "",
        "```bash",
        f"python -m financebench.cli eval --benchmark {inputs.benchmark_or_group} "
        f"--model-config <config used> --seed {inputs.config.seed} "
        f"--prompt-profile {inputs.config.prompt_profile} "
        f"--eval-mode {inputs.config.eval_mode.value}",
        "```",
        "",
        "_A benchmark measures a benchmark. It cannot tell you what this model does on data it has "
        "never seen, in a workflow it was not tested in._",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _scores_html(inputs: ArtifactInputs) -> str:
    scores = inputs.scores
    if scores is None:
        return ""
    items = []
    for label, value in (
        ("Financial Core Score (context_given)", scores.core_score),
        ("Financial RAG Score (retrieval_required)", scores.rag_score),
        ("Financial Agent Score (tool_assisted)", scores.agent_score),
    ):
        items.append(
            f"<li>{html.escape(label)}: <strong>{value:.3f}</strong></li>"
            if value is not None
            else f"<li>{html.escape(label)}: <em>not measured</em></li>"
        )
    if scores.fci is not None:
        items.append(f"<li><strong>Finance Capability Index: {scores.fci:.3f}</strong></li>")
    else:
        items.append(
            "<li>Finance Capability Index: <strong>withheld</strong> — "
            f"{html.escape(scores.fci_withheld_because or '')}</li>"
        )
    return f"<h3>Top-level scores</h3><ul>{''.join(items)}</ul>"


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

    def _gate_row(gate: GateResult) -> str:
        # A gate the run had nothing to test with is NOT a failing gate, and it is not a passing one
        # either. Rendering `passed=None` as FAIL would invent a defect; rendering it as PASS would
        # invent a guarantee. It gets its own word.
        if gate.skipped:
            status, css = "NOT TESTED", "skip"
        elif gate.passed:
            status, css = "PASS", "pass"
        else:
            status, css = "FAIL", "fail"
        observed = "&mdash;" if gate.observed is None else gate.observed
        return (
            f"<tr><td>{esc(gate.gate_name)}</td><td>{gate.threshold}</td><td>{observed}</td>"
            f"<td class='{css}'>{status}</td></tr>"
        )

    gate_rows = (
        "\n".join(_gate_row(g) for g in inputs.gates.gates)
        or "<tr><td colspan='4'>Gates not evaluated (no samples scored).</td></tr>"
    )
    distribution = failure_distribution(inputs.failures)
    failure_rows = (
        "\n".join(
            f"<tr><td>{esc(name)}</td><td>{count}</td></tr>" for name, count in distribution.items()
        )
        or "<tr><td colspan='2'>No failures.</td></tr>"
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
  .verdict {{ font-size: 1.3rem; padding: 0.75rem 1rem; background: #f0f9ff; border-left: 5px solid #0369a1; }}
  .verdict code {{ background: none; font-weight: 700; }}
  td.pass {{ color: #166534; font-weight: 600; }}
  td.skip {{ opacity: .65; font-weight: 600; }}
  td.fail {{ color: #b91c1c; font-weight: 700; }}
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

  <h2>Verdict</h2>
  <div class="verdict"><code>{esc(inputs.verdict)}</code></div>
  <ul>{"".join(f"<li>{esc(r)}</li>" for r in inputs.verdict_reasons)}</ul>
  {_scores_html(inputs)}

  <h2>Critical gates</h2>
  <p class="note">Evaluated independently of the average. A failed critical gate caps the verdict
     no matter how good the mean looks — because being off by 1000x is not a near-miss.</p>
  <table>
    <tr><th>Gate</th><th>Threshold</th><th>Observed</th><th>Result</th></tr>
    {gate_rows}
  </table>

  <h2>How it failed</h2>
  <table>
    <tr><th>Failure type</th><th>Count</th></tr>
    {failure_rows}
  </table>

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
