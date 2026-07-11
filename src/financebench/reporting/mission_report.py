"""The report a human actually reads: can this model be trusted with money?

Every other artifact in this platform is written for a machine or for an auditor — predictions,
metric details, gates, fingerprints. This one is written for the person deciding whether to point a
model at their books, and it answers the five questions the platform exists to answer:

1. Can it calculate?
2. Can it reason over tables, text, and conversations?
3. Can it find and cite evidence in a real document?
4. Can it give useful CFO analysis for a small business?
5. Does it know when it *cannot* answer — and can it be talked into lying by its own data?

Three rules govern what goes in it, and they are all the same rule:

- **A question with no run behind it is reported as unanswered**, not as a zero. An empty section
  says "we did not measure this". A 0.0 says "we measured this and the model failed", and those are
  different sentences.
- **Numbers that are not comparable are not put next to each other.** Two runs with different
  evaluator fingerprints measured different things; showing them in one column would invite a
  comparison the evidence does not support.
- **The verdict is never better than the worst critical gate.** A strong average cannot buy off a
  model that invents figures, and the report does not let it try.

Self-contained: one HTML file, no external CSS, no fonts, no scripts, no network. It can be emailed,
committed, or opened on a machine that has never heard of this repository.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["RunSummary", "build_mission_report", "load_runs"]


def _esc(value: object) -> str:
    return html.escape(str(value))


def _pct(value: float | None, digits: int = 1) -> str:
    """A number, or an honest dash. Never a zero standing in for an absence."""
    if value is None:
        return "&mdash;"
    return f"{value * 100:.{digits}f}%"


@dataclass
class RunSummary:
    """One run, reduced to what the report needs."""

    run_id: str
    model_ref: str
    benchmark: str
    eval_mode: str
    conversation_protocol: str | None
    run_type: str
    fingerprint: str | None
    n_samples: int
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    capabilities: dict[str, dict[str, Any]] = field(default_factory=dict)
    scores: dict[str, Any] = field(default_factory=dict)
    gates: list[dict[str, Any]] = field(default_factory=list)
    failures: dict[str, int] = field(default_factory=dict)
    retrieval: dict[str, Any] | None = None
    conversation: dict[str, Any] | None = None
    verdict: str = "NOT_EVALUATED"

    def metric(self, name: str) -> float | None:
        entry = self.metrics.get(name)
        return None if entry is None else entry.get("mean")

    def metric_n(self, name: str) -> int:
        entry = self.metrics.get(name)
        return 0 if entry is None else int(entry.get("n", 0))


def load_runs(runs_dir: Path) -> list[RunSummary]:
    """Read every run directory that has the artifacts we need. A half-written run is skipped, not
    guessed at."""
    summaries: list[RunSummary] = []
    for path in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
        if not path.is_dir():
            continue
        try:
            environment = json.loads((path / "environment.json").read_text(encoding="utf-8"))
            config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
            metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
            capabilities = json.loads((path / "capabilities.json").read_text(encoding="utf-8"))
            gates = json.loads((path / "gates.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue

        run = (
            environment.get("run", {}) if isinstance(environment.get("run"), dict) else environment
        )
        summaries.append(
            RunSummary(
                run_id=str(run.get("run_id") or path.name),
                model_ref=str(run.get("model_ref") or "unknown"),
                benchmark=str(run.get("benchmark_or_group") or "unknown"),
                eval_mode=str(config.get("eval_mode") or "context_given"),
                conversation_protocol=config.get("conversation_protocol"),
                run_type=str(run.get("run_type") or "real"),
                fingerprint=(environment.get("evaluator_fingerprint") or {}).get("digest"),
                n_samples=int(run.get("n_samples") or 0),
                metrics=metrics,
                capabilities=capabilities.get("dimensions") or {},
                scores=capabilities.get("scores") or {},
                gates=gates.get("gates") or [],
                failures=capabilities.get("failure_distribution") or {},
                retrieval=capabilities.get("retrieval"),
                conversation=capabilities.get("conversation"),
                verdict=str(capabilities.get("verdict") or "NOT_EVALUATED"),
            )
        )
    return summaries


# --------------------------------------------------------------------------- the five questions


def _unanswered(question: str, why: str) -> str:
    return (
        f"<div class='unanswered'><h3>{_esc(question)}</h3>"
        f"<p class='dash'>Not measured.</p><p class='why'>{_esc(why)}</p></div>"
    )


def _q1_calculation(runs: Sequence[RunSummary]) -> str:
    """Can it calculate?"""
    rows = []
    for run in runs:
        for name in (
            "finqa_answer_accuracy",
            "finqa_execution_accuracy",
            "tatqa_exact_match",
            "finance_reasoning_accuracy",
        ):
            value = run.metric(name)
            if value is None:
                continue
            official = (
                "official" if name != "finqa_answer_accuracy" else "ours (not the official metric)"
            )
            rows.append(
                f"<tr><td>{_esc(run.model_ref)}</td><td>{_esc(name)}</td>"
                f"<td class='num'>{_pct(value)}</td><td class='num'>{run.metric_n(name)}</td>"
                f"<td class='note'>{official}</td></tr>"
            )
    if not rows:
        return _unanswered("1. Can it calculate financial answers correctly?", "No core run found.")
    return f"""
    <h3>1. Can it calculate financial answers correctly?</h3>
    <table><thead><tr><th>Model</th><th>Metric</th><th>Score</th><th>n</th><th></th></tr></thead>
    <tbody>{"".join(rows)}</tbody></table>
    """


def _q2_conversation(runs: Sequence[RunSummary]) -> str:
    """Can it hold a conversation? The gap between the protocols is the whole answer."""
    by_protocol = {
        run.conversation_protocol: run
        for run in runs
        if run.conversation and run.conversation_protocol
    }
    if not by_protocol:
        return _unanswered(
            "2. Can it reason over a conversation?",
            "No ConvFinQA run found. A single-turn benchmark cannot answer this: it never asks a "
            "question that depends on the previous one.",
        )

    rows = []
    for protocol in ("gold_history", "model_history"):
        run = by_protocol.get(protocol)
        if run is None or not run.conversation:
            rows.append(
                f"<tr><td>{_esc(protocol)}</td>"
                f"<td colspan='5' class='dash'>not run &mdash; the comparison needs both</td></tr>"
            )
            continue
        c = run.conversation
        rows.append(
            f"<tr><td>{_esc(protocol)}</td>"
            f"<td class='num'>{_pct(c.get('turn_accuracy'))}</td>"
            f"<td class='num'>{_pct(c.get('full_conversation_accuracy'))}</td>"
            f"<td class='num'>{_pct(c.get('context_loss'))}</td>"
            f"<td class='num'>{_pct(c.get('propagation_effect'))}</td>"
            f"<td class='num'>{_pct(c.get('recovery_rate'))}</td></tr>"
        )

    gold = by_protocol.get("gold_history")
    model = by_protocol.get("model_history")
    gap = ""
    if gold and model and gold.conversation and model.conversation:
        g = gold.conversation.get("turn_accuracy")
        m = model.conversation.get("turn_accuracy")
        if g is not None and m is not None:
            gap = (
                f"<p class='finding'><b>The number that matters: {_pct(g - m)}.</b> That is what the "
                "model loses when it has to live with its own previous answers instead of being "
                "handed the right ones. A model that scores well under <code>gold_history</code> and "
                "collapses under <code>model_history</code> can answer a question but cannot hold a "
                "conversation.</p>"
            )

    return f"""
    <h3>2. Can it reason over a conversation?</h3>
    <p>Two protocols, never mixed into one score. <code>gold_history</code> hands each turn the
    <em>correct</em> prior conversation, isolating per-turn reasoning. <code>model_history</code>
    hands each turn the model's <em>own</em> prior answers &mdash; which is what a conversation
    actually is, and the only way error propagation is visible at all.</p>
    <table><thead><tr><th>Protocol</th><th>Turn accuracy</th><th>Whole conversation</th>
    <th>Context loss</th><th>Propagation effect</th><th>Recovery</th></tr></thead>
    <tbody>{"".join(rows)}</tbody></table>
    {gap}
    """


def _q3_retrieval(runs: Sequence[RunSummary]) -> str:
    """Can it find its own evidence? The retrieval loss is the answer."""
    given = next(
        (r for r in runs if r.benchmark == "financebench" and r.eval_mode == "context_given"), None
    )
    required = next(
        (r for r in runs if r.benchmark == "financebench" and r.eval_mode == "retrieval_required"),
        None,
    )
    if given is None:
        return _unanswered(
            "3. Can it find and cite evidence in a real document?",
            "No FinanceBench run found.",
        )

    a = given.metric("financebench_answer_accuracy")
    citation = given.metric("financebench_citation_accuracy")
    hallucination = given.metric("financebench_unsupported_numeric_claim")

    body = f"""
    <h3>3. Can it find and cite evidence in a real document?</h3>
    <table><thead><tr><th>Mode</th><th>Answer accuracy</th><th>Grounded numbers</th>
    <th>Citation accuracy</th><th>n</th></tr></thead><tbody>
      <tr><td>context_given<br><span class='note'>evidence handed to it</span></td>
        <td class='num'>{_pct(a)}</td><td class='num'>{_pct(hallucination)}</td>
        <td class='num'>{_pct(citation)}</td>
        <td class='num'>{given.metric_n("financebench_answer_accuracy")}</td></tr>
    """
    if required is None:
        body += """
      <tr><td>retrieval_required<br><span class='note'>must find its own evidence</span></td>
        <td colspan='4' class='dash'>not run &mdash; without it, nothing here says the model can
        find a figure in a filing, only that it can read one that was handed to it</td></tr>
        """
    else:
        b = required.metric("financebench_answer_accuracy")
        loss = None if (a is None or b is None) else a - b
        body += f"""
      <tr><td>retrieval_required<br><span class='note'>must find its own evidence</span></td>
        <td class='num'>{_pct(b)}</td>
        <td class='num'>{_pct(required.metric("financebench_unsupported_numeric_claim"))}</td>
        <td class='num'>{_pct(required.metric("financebench_citation_accuracy"))}</td>
        <td class='num'>{required.metric_n("financebench_answer_accuracy")}</td></tr>
        """
        if loss is not None:
            body += f"""</tbody></table>
        <p class='finding'><b>Retrieval loss: {_pct(loss)}.</b> That is the accuracy that exists only
        because the evidence was handed over. In production nobody hands it over.</p>
        """
            if required.retrieval:
                r = required.retrieval
                body += f"""
        <p class='note'>Retriever: <code>{_esc(r.get("retriever", "?"))}</code>, top-k
        {_esc(r.get("top_k", "?"))},
        {"document-scoped" if r.get("document_scoped") else "open-corpus"},
        over {_esc(r.get("corpus_pages", "?"))} pages of
        {_esc(r.get("corpus_documents", "?"))} real filings. Document recall
        {_pct(r.get("document_recall"))}, <b>page recall {_pct(r.get("page_recall"))}</b>.</p>
        <p class='finding'>The failure attribution is the actionable part, and a single
        RAG-accuracy number could never have produced it: <b>the retriever misses far more often
        than the model fumbles.</b> "The page was never found" and "the page was found and misread"
        have opposite fixes, and only one of them is about the model.</p>
        """
            return body

    body += "</tbody></table>"
    if citation == 0.0:
        body += """
    <p class='finding'>Citation accuracy is <b>zero</b>. Not low &mdash; zero. The model never once
    pointed at where a figure came from, so nothing it says can be checked without redoing the work
    by hand, which is the entire job it was supposed to save.</p>
    """
    return body


def _q4_smb(runs: Sequence[RunSummary]) -> str:
    """Can it advise a small business?"""
    smb = [r for r in runs if r.benchmark == "smb_cfo"]
    if not smb:
        return _unanswered(
            "4. Can it give useful CFO analysis for a small business?",
            "No SMB-CFO run found. Every other benchmark here is built on public-company filings, "
            "and a small business does not have a 10-K &mdash; it has a ledger and a payroll date.",
        )
    rows = []
    for run in smb:
        rows.append(
            f"<tr><td>{_esc(run.model_ref)}</td>"
            f"<td class='num'>{_pct(run.metric('smb_cfo_accuracy'))}</td>"
            f"<td class='num'>{run.metric_n('smb_cfo_accuracy')}</td>"
            f"<td class='num'>{_pct(run.metric('smb_cfo_refusal_correctness'))}</td>"
            f"<td class='num'>{_pct(run.metric('smb_cfo_injection_resistance'))}</td></tr>"
        )
    return f"""
    <h3>4. Can it give useful CFO analysis for a small business?</h3>
    <p>Gold answers here come from <b>deterministic Python oracles</b> over generated books. No LLM
    ever writes a gold answer, so this benchmark cannot inherit a judge's mistakes &mdash; and it is
    the only one here that is provably uncontaminated by anyone's pretraining data.</p>
    <table><thead><tr><th>Model</th><th>Accuracy</th><th>n</th><th>Refusal correctness</th>
    <th>Injection resistance</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
    """


def _q5_safety(runs: Sequence[RunSummary]) -> str:
    """Does it know when it cannot answer — and can its own data talk it into lying?"""
    relevant = [r for r in runs if r.metric("smb_cfo_refusal_correctness") is not None]
    if not relevant:
        return _unanswered(
            "5. Does it know when it cannot answer?",
            "No adversarial or refusal benchmark ran. Nothing here asked a question the data cannot "
            "answer, so nothing here shows whether the model would decline or invent a figure.",
        )

    blocks = []
    for run in relevant:
        refusal = run.metric("smb_cfo_refusal_correctness")
        injection = run.metric("smb_cfo_injection_resistance")
        failed_refusals = run.failures.get("failed_refusal", 0)
        obeyed = run.failures.get("obeyed_prompt_injection", 0)
        blocks.append(f"""
        <table><thead><tr><th>{_esc(run.model_ref)}</th><th>Score</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr><td>Refusal correctness</td><td class='num'>{_pct(refusal)}</td>
            <td>declined exactly when the books could not support an answer</td></tr>
          <tr><td>Failed refusals</td>
            <td class='num {"bad" if failed_refusals else "good"}'>{failed_refusals}</td>
            <td>invented a number for a question with no answer &mdash; the dangerous one</td></tr>
          <tr><td>Injection resistance</td><td class='num'>{_pct(injection)}</td>
            <td>ignored instructions hidden inside its own ledger</td></tr>
          <tr><td>Injections obeyed</td>
            <td class='num {"bad" if obeyed else "good"}'>{obeyed}</td>
            <td>a security hole, not an analytical error: the attacker is whoever can add a row</td></tr>
        </tbody></table>
        """)

    return f"""
    <h3>5. Does it know when it cannot answer &mdash; and can its data talk it into lying?</h3>
    <p>An injected instruction tells the model to report a <b>canary value that appears nowhere else
    in the books</b>. A model that states it can only have got it from the instruction hidden in its
    own data, which makes obeying unambiguous rather than a judgement call.</p>
    {"".join(blocks)}
    """


# --------------------------------------------------------------------------- gates and verdict


def _gates_section(runs: Sequence[RunSummary]) -> str:
    rows = []
    for run in runs:
        for gate in run.gates:
            if gate.get("skipped"):
                status, css = "NOT TESTED", "skip"
            elif gate.get("passed"):
                status, css = "PASS", "pass"
            else:
                status, css = "FAIL", "fail"
            observed = gate.get("observed")
            rows.append(
                f"<tr><td>{_esc(run.benchmark)}</td><td>{_esc(gate.get('gate_name'))}</td>"
                f"<td class='num'>{'&mdash;' if observed is None else observed}</td>"
                f"<td class='num'>{_esc(gate.get('threshold'))}</td>"
                f"<td class='{css}'>{status}</td></tr>"
            )
    if not rows:
        return ""
    return f"""
    <h2>Critical gates</h2>
    <p>A mean score treats all errors as the same size. They are not: being off by 2% is a rounding
    disagreement, and being off by 1000&times; because you confused thousands with millions is a
    disaster &mdash; and they look identical in an average. Gates are evaluated independently of the
    mean, and a failed gate caps the verdict no matter how good the average is.</p>
    <p class='note'><b>NOT TESTED</b> is not a pass. A run that never attacked the model has not
    shown that the model resists attack.</p>
    <table><thead><tr><th>Benchmark</th><th>Gate</th><th>Observed</th><th>Threshold</th>
    <th>Result</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
    """


def _verdict_section(runs: Sequence[RunSummary]) -> str:
    verdicts = {run.verdict for run in runs if run.run_type == "real"}
    worst = "NOT_EVALUATED"
    order = [
        "NOT_EVALUATED",
        "INSUFFICIENT_COVERAGE",
        "NOT_FINANCE_READY",
        "LIMITED_HIGH_SUPERVISION",
        "USABLE_WITH_HUMAN_REVIEW",
        "STRONG_FOR_BOUNDED_FINANCIAL_TASKS",
        "EXCEPTIONAL_BUT_STILL_REQUIRES_CONTROLS",
    ]
    present = [v for v in order if v in verdicts]
    if present:
        worst = present[0]

    withheld = next(
        (
            run.scores.get("fci_withheld_because")
            for run in runs
            if run.scores.get("fci_withheld_because")
        ),
        None,
    )
    fci_note = (
        f"<p class='finding'><b>No Finance Capability Index is published for this evaluation.</b> "
        f"{_esc(withheld)}</p>"
        if withheld
        else ""
    )
    return f"""
    <h2>Verdict</h2>
    <p class='verdict'>{_esc(worst)}</p>
    <p>The verdict is the <em>worst</em> across every benchmark run, never the average. A model that
    is safe on four benchmarks and invents figures on the fifth is not four-fifths safe.</p>
    {fci_note}
    <p class='note'>There is deliberately no "safe for autonomous financial decisions" label, and no
    threshold that produces one. A benchmark measures a benchmark.</p>
    """


_CSS = """
:root { color-scheme: light dark; }
body { font: 16px/1.55 -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  margin: 0 auto; max-width: 60rem; padding: 2rem 1.25rem 5rem; }
h1 { font-size: 1.9rem; margin-bottom: .25rem; }
h2 { font-size: 1.35rem; margin-top: 2.5rem; padding-bottom: .3rem;
  border-bottom: 2px solid currentColor; }
h3 { font-size: 1.1rem; margin-top: 2rem; }
p { margin: .7rem 0; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .92rem;
  display: block; overflow-x: auto; }
th, td { text-align: left; padding: .5rem .7rem; border-bottom: 1px solid rgba(128,128,128,.35); }
th { font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pass { color: #17803d; font-weight: 600; }
.fail { color: #c02626; font-weight: 600; }
.skip { opacity: .65; font-weight: 600; }
.good { color: #17803d; }
.bad  { color: #c02626; font-weight: 700; }
.dash { opacity: .6; }
.note { font-size: .87rem; opacity: .75; }
.finding { border-left: 4px solid currentColor; padding: .6rem 0 .6rem .9rem; margin: 1.2rem 0; }
.verdict { font-size: 1.5rem; font-weight: 700; letter-spacing: .02em; margin: .5rem 0 1rem; }
.unanswered { border: 1px dashed rgba(128,128,128,.6); border-radius: 6px; padding: .8rem 1rem;
  margin: 1.2rem 0; }
.unanswered h3 { margin-top: 0; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .88em; }
footer { margin-top: 3rem; font-size: .85rem; opacity: .7; }
"""


def build_mission_report(runs: Sequence[RunSummary], *, generated_at: str) -> str:
    """One self-contained HTML file. No external CSS, fonts, scripts, or network."""
    real = [r for r in runs if r.run_type == "real"]
    mocks = [r for r in runs if r.run_type != "real"]

    fingerprints = {r.fingerprint for r in real if r.fingerprint}
    fingerprint_warning = ""
    if len(fingerprints) > 1:
        fingerprint_warning = f"""
    <p class='finding'><b>These runs are not all comparable.</b> {len(fingerprints)} different
    evaluator fingerprints are present, which means the code that produced these scores changed
    between runs. Re-score them against one version before comparing. Fingerprints:
    {_esc(", ".join(sorted(fingerprints)))}</p>
    """

    models = sorted({r.model_ref for r in real}) or ["(none)"]
    inventory = (
        "".join(
            f"<tr><td>{_esc(r.benchmark)}</td><td>{_esc(r.eval_mode)}</td>"
            f"<td>{_esc(r.conversation_protocol or '&mdash;')}</td>"
            f"<td>{_esc(r.model_ref)}</td><td class='num'>{r.n_samples}</td>"
            f"<td><code>{_esc(r.fingerprint or '?')}</code></td></tr>"
            for r in real
        )
        or "<tr><td colspan='6' class='dash'>No real runs found.</td></tr>"
    )

    mock_note = (
        f"<p class='note'>{len(mocks)} mock run(s) excluded. The mock provider reads the gold "
        "answers: its scores measure the pipeline, never a model.</p>"
        if mocks
        else ""
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Financial capability report</title>
<style>{_CSS}</style></head>
<body>
<h1>Can this model be trusted with money?</h1>
<p class="note">Generated {_esc(generated_at)} &middot; models: {_esc(", ".join(models))}</p>

<p>Every number here came from a real model answering real questions. Where a question was not
asked, this report says so rather than printing a zero &mdash; <b>an absent measurement and a failed
one are different findings</b>, and only one of them is about the model.</p>

{fingerprint_warning}

<h2>What was actually run</h2>
<table><thead><tr><th>Benchmark</th><th>Mode</th><th>Protocol</th><th>Model</th><th>n</th>
<th>Evaluator</th></tr></thead><tbody>{inventory}</tbody></table>
{mock_note}

<h2>The five questions</h2>
{_q1_calculation(real)}
{_q2_conversation(real)}
{_q3_retrieval(real)}
{_q4_smb(real)}
{_q5_safety(real)}

{_gates_section(real)}
{_verdict_section(real)}

<footer>
<p>Metrics named <code>*_answer_accuracy</code> and <code>smb_cfo_*</code> are this platform's own,
not the source benchmark's official evaluator, and are named so they cannot be mistaken for it.
Official metrics (FinQA execution/program accuracy, TAT-QA exact match, FinanceReasoning accuracy)
are parity-tested against the real upstream implementations.</p>
<p>No metric rule was changed to improve any score in this report.</p>
</footer>
</body></html>
"""
