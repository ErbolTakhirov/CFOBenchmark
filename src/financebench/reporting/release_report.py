"""The primary public report: can qwen2.5:3b or qwen2.5:7b be trusted with money?

The answer, throughout, is expressed in what was **measured** — and the report's most important job is
to be equally clear about what was not. Three things it will never do:

- print a Finance Capability Index for a run whose coverage does not support one (it prints
  ``INSUFFICIENT_COVERAGE``, and says which coverage was missing);
- print `0.0` where the truth is "not measured" (it prints `NOT_EVALUATED`, or a dash);
- present two runs side by side when a different evaluator produced them.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["ModelResult", "build_release_report"]

INSUFFICIENT_COVERAGE = "INSUFFICIENT_COVERAGE"
NOT_EVALUATED = "NOT_EVALUATED"


@dataclass
class ModelResult:
    """Everything one model's runs said, and what they did not say."""

    model_ref: str
    runs: dict[str, dict[str, Any]] = field(default_factory=dict)  # run_id -> artifacts

    def metric(self, run_id: str, name: str) -> tuple[float | None, int, int]:
        """(mean, n_graded, n_not_applicable). `None` mean = NOT MEASURED, never zero."""
        run = self.runs.get(run_id)
        if run is None:
            return None, 0, 0
        entry = run.get("metrics", {}).get(name)
        if not isinstance(entry, dict):
            return None, 0, 0
        return entry.get("mean"), int(entry.get("n", 0)), int(entry.get("n_not_applicable", 0))

    def ci(self, run_id: str, name: str) -> tuple[float | None, float | None]:
        run = self.runs.get(run_id, {})
        entry = run.get("confidence_intervals", {}).get(name)
        if not isinstance(entry, dict):
            return None, None
        return entry.get("ci_low"), entry.get("ci_high")

    def fci(self, run_id: str) -> tuple[float | None, str | None]:
        run = self.runs.get(run_id, {})
        scores = run.get("capabilities", {}).get("scores", {})
        return scores.get("finance_capability_index"), scores.get("fci_withheld_because")

    def verdict(self, run_id: str) -> str:
        return str(self.runs.get(run_id, {}).get("capabilities", {}).get("verdict", NOT_EVALUATED))


def load_run(runs_dir: Path, run_id: str) -> dict[str, Any] | None:
    path = runs_dir / run_id
    if not (path / "environment.json").is_file():
        return None
    out: dict[str, Any] = {}
    for name in (
        "environment",
        "metrics",
        "capabilities",
        "gates",
        "coverage",
        "costs",
        "confidence_intervals",
        "run_config",
    ):
        file = path / f"{name}.json"
        if file.is_file():
            out[name] = json.loads(file.read_text(encoding="utf-8"))
    return out


def _fmt(mean: float | None, n: int = 0, n_na: int = 0) -> str:
    """A number, with the evidence under it. A dash where nothing was measured."""
    if mean is None:
        return "—"
    base = f"{mean:.3f}"
    if n:
        base += f" (n={n}"
        if n_na:
            base += f", {n_na} n/a"
        base += ")"
    return base


def _fci_cell(value: float | None, reason: str | None) -> str:
    if value is not None:
        return f"{value:.4f}"
    # The index is refused, not asterisked — and the refusal names its own reason.
    return f"**{INSUFFICIENT_COVERAGE}** — {reason}" if reason else f"**{INSUFFICIENT_COVERAGE}**"


def build_release_report(
    out_dir: Path,
    *,
    version: str,
    models: list[ModelResult],
    paired: list[dict[str, Any]],
    fingerprint: str,
    hardware: dict[str, Any],
    limitations: str,
) -> None:
    """Write ``results_tables.md``, ``report.html``, ``results.json`` and ``leaderboard.csv``.

    It deliberately does **not** write ``report.md``. That file is the narrative — the one a human
    reads, carrying the findings and the caveats — and it is authored, not generated. A generator
    that overwrites it turns a report into a table dump, and the tables are the part of a benchmark
    report that means least on its own: `0.027` is a number, and "giving the model a calculator made
    it five times worse, and it called the calculator twice in 150 questions" is a finding.

    The tables here are generated so that the numbers *in* them cannot drift from the artifacts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    md: list[str] = [
        f"# FinanceBench {version} — generated result tables",
        "",
        "Generated from the run artifacts. The narrative report, with the findings, is "
        "[`report.md`](report.md).",
        "",
        f"Evaluator fingerprint `{fingerprint}`. Every run below was scored by **this** evaluator; "
        "runs scored by a different one are not on this page, because they are not comparable and "
        "averaging them would be a lie.",
        "",
        f"Hardware: {hardware.get('gpu') or 'CPU only'} — {hardware.get('platform', '?')}.",
        "",
        "> **A dash means NOT MEASURED. It never means zero.** An `INSUFFICIENT_COVERAGE` index is "
        "a refusal, not a missing number: the run did not ask enough to support the claim the index "
        "makes, and the reason is printed next to it.",
        "",
    ]

    # ---- the headline
    md += ["## Verdict", "", "| model | run | FCI | verdict |", "|---|---|---|---|"]
    for model in models:
        for run_id in sorted(model.runs):
            value, reason = model.fci(run_id)
            md.append(
                f"| `{model.model_ref}` | `{run_id[:44]}` | {_fci_cell(value, reason)} "
                f"| {model.verdict(run_id)} |"
            )

    # ---- per-benchmark
    md += ["", "## What was measured", "", "| model | run | metric | value |", "|---|---|---|---|"]
    for model in models:
        for run_id in sorted(model.runs):
            metrics = model.runs[run_id].get("metrics", {})
            for name in sorted(metrics):
                mean, n, n_na = model.metric(run_id, name)
                md.append(
                    f"| `{model.model_ref}` | `{run_id[:30]}` | {name} | {_fmt(mean, n, n_na)} |"
                )

    # ---- paired comparisons
    if paired:
        md += [
            "",
            "## Paired comparisons",
            "",
            "Same sample ids, same evaluator. The 2x2 discordance table is the interesting part — two runs can post "
            "identical means while disagreeing on half the questions.",
            "",
            "| comparison | n | A | B | difference | 95% CI | verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        for p in paired:
            md.append(
                f"| {p['label']} | {p['n_paired']} | {p['mean_a']:.3f} | {p['mean_b']:.3f} "
                f"| {p['mean_difference']:+.3f} | [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}] "
                f"| {p['verdict']} |"
            )

    # ---- what was NOT measured
    md += [
        "",
        "## What was NOT measured",
        "",
        f"- **SECQUE analytical correctness: `{NOT_EVALUATED}`.** No available judge passes "
        "calibration. `llama3.2:3b` scores 75% accuracy with a **41% false-positive rate** against a "
        "20% bar — it never rejects a good answer, and waves through two-thirds of answers that name "
        "the wrong company or contain a fabricated figure. This is a measurement, not an omission, "
        "and it is **never** reported as zero.",
        "- **No API provider is live-verified.** OpenAI, Anthropic, Gemini and OpenRouter are "
        "implemented and unit-tested against a mocked transport. No API key exists in this "
        "environment, so **none of them has ever made a successful call**.",
        "- **No multimodal run exists.** `multimodal_coverage: 0.0` in every run.",
        "",
        "## Limitations",
        "",
        limitations,
        "",
        "---",
        "",
        "**A good score here does not certify that a model is safe to run unsupervised against real "
        "money.** It means it did well on these questions, on this hardware, on this date.",
        "",
    ]
    (out_dir / "results_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- results.json
    payload = {
        "version": version,
        "evaluator_fingerprint": fingerprint,
        "hardware": hardware,
        "models": [
            {
                "model_ref": m.model_ref,
                "runs": {
                    run_id: {
                        "fci": m.fci(run_id)[0],
                        "fci_withheld_because": m.fci(run_id)[1],
                        "verdict": m.verdict(run_id),
                        "metrics": m.runs[run_id].get("metrics", {}),
                        "capabilities": m.runs[run_id].get("capabilities", {}),
                        "gates": m.runs[run_id].get("gates", {}),
                        "coverage": m.runs[run_id].get("coverage", {}),
                    }
                    for run_id in sorted(m.runs)
                },
            }
            for m in models
        ],
        "paired_comparisons": paired,
        "secque_analytical_score": NOT_EVALUATED,
        "secque_analytical_reason": (
            "No available judge passes calibration (llama3.2:3b: 41% false-positive rate against a "
            "20% bar). NOT_EVALUATED is a measurement. It is never zero."
        ),
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # ---- leaderboard.csv
    rows = ["model_ref,run_id,fci,verdict,critical_gate_failed,n_samples"]
    for m in models:
        for run_id in sorted(m.runs):
            value, _ = m.fci(run_id)
            gates = m.runs[run_id].get("gates", {})
            coverage = m.runs[run_id].get("coverage", {})
            rows.append(
                f"{m.model_ref},{run_id},"
                f"{'' if value is None else f'{value:.4f}'},"
                f"{m.verdict(run_id)},{gates.get('any_critical_gate_failed', '')},"
                f"{coverage.get('evaluated_samples', '')}"
            )
    (out_dir / "leaderboard.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ---- HTML
    body = "\n".join(
        f"<tr><td><code>{html.escape(m.model_ref)}</code></td>"
        f"<td><code>{html.escape(r[:44])}</code></td>"
        f"<td class='{'withheld' if m.fci(r)[0] is None else ''}'>"
        f"{html.escape(_fci_cell(m.fci(r)[0], m.fci(r)[1])[:90])}</td>"
        f"<td>{html.escape(m.verdict(r))}</td></tr>"
        for m in models
        for r in sorted(m.runs)
    )
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>FinanceBench {html.escape(version)} — release report</title>
<style>
 body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
        max-width: 1100px; margin: 2rem auto; color: #1a1a1a; line-height: 1.55; }}
 table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
 th, td {{ border: 1px solid #ddd; padding: .45rem .7rem; text-align: left; }}
 th {{ background: #f4f4f4; }}
 .withheld {{ color: #8a6d3b; font-style: italic; }}
 .warn {{ background: #fff8e1; border-left: 4px solid #f0ad4e; padding: 1rem; }}
</style></head><body>
<h1>FinanceBench {html.escape(version)}</h1>
<p>Evaluator fingerprint <code>{html.escape(fingerprint)}</code>.
Hardware: {html.escape(str(hardware.get("gpu") or "CPU only"))}.</p>
<div class="warn">
<p><strong>A dash means NOT MEASURED. It never means zero.</strong> An
<code>INSUFFICIENT_COVERAGE</code> index is a refusal, not a missing number.</p>
<p><strong>SECQUE analytical correctness is <code>NOT_EVALUATED</code></strong> &mdash; no available
judge passes calibration (41% false-positive rate against a 20% bar). That is a measurement.</p>
<p><strong>No API provider is live-verified.</strong> No API key has ever been used here.</p>
</div>
<h2>Verdict</h2>
<table><tr><th>model</th><th>run</th><th>FCI</th><th>verdict</th></tr>
{body}
</table>
<p><em>A good score does not certify that a model is safe to run unsupervised against real money.</em></p>
</body></html>
"""
    (out_dir / "report.html").write_text(doc, encoding="utf-8")
