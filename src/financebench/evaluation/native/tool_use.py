"""Tool-use metrics. The question is not "did it get the answer right" — accuracy already asks that.

The question is **where in the chain it broke**, because "the model is bad at tools" is four
completely different defects with four different fixes:

- it did not reach for a tool at all (a planner failure);
- it reached for a tool that does not exist (a hallucination);
- it reached for the right tool with malformed arguments (a schema failure);
- it called the right tool correctly, got the right number back, **and then ignored it**.

That last one is the reason this file exists. A model can call the calculator, receive ``40.55``, and
write "approximately 38". Accuracy records a failure and attributes it to arithmetic — which is
precisely the thing the model got *right*. The defect is that it did not believe its own tool, and no
end-to-end metric can see it. Only the trace can.

It also answers the question the whole tool benchmark is *for*: **does giving a model tools actually
help?** The honest answer is not obviously yes. Tools add an orchestration surface that a small model
can fail at, and a model that would have guessed the right number can instead fail to format a JSON
tool call and produce nothing at all. The paired direct-vs-tools comparison measures whether the trade
is worth it, and it is entirely permitted to come out negative.
"""

from __future__ import annotations

from financebench.evaluation.metrics.base import Metric, register_metric
from financebench.schemas.metric import MetricResult
from financebench.schemas.prediction import Prediction
from financebench.schemas.sample import CanonicalSample
from financebench.tools.trace import ToolTrace

__all__ = [
    "TOOL_TRACES",
    "ToolExecutionSuccess",
    "ToolResultUtilization",
    "ToolSecurityRejection",
    "ToolSelectionAccuracy",
    "register_trace",
]

#: sample_id -> trace. Populated by the runner, read by the metrics.
#:
#: A module-level side-channel is not elegant, and it is here for a reason worth stating: the
#: `Metric` contract is `(sample, prediction) -> result`, and a `Prediction` has nowhere to put a
#: tool trace. Widening the schema for every benchmark to carry a field only one of them uses would
#: put an empty `tool_trace: null` into every FinQA prediction ever written. The traces are also
#: persisted to `tool_traces.jsonl` in the run artifacts, which is the durable record; this dict is
#: just how the metrics reach them in-process.
TOOL_TRACES: dict[str, ToolTrace] = {}


def register_trace(trace: ToolTrace) -> None:
    TOOL_TRACES[trace.sample_id] = trace


def _trace(sample: CanonicalSample) -> ToolTrace | None:
    return TOOL_TRACES.get(sample.sample_id)


def _no_trace(sample: CanonicalSample, metric: str) -> MetricResult:
    """This run had no tools. Not a failure — a different experiment entirely."""
    return MetricResult(
        sample_id=sample.sample_id,
        metric_name=metric,
        value=None,
        passed=None,
        details={"reason": "no tool trace — this was not a tool-assisted run"},
    )


@register_metric("tool_selection_accuracy")
class ToolSelectionAccuracy(Metric):
    """Did it reach for a real tool at all?

    ``passed=False`` covers two distinct sins, and the details say which: the model called **no**
    tool on a question that needs arithmetic (it did the sums in its head), or it called a tool that
    **does not exist** (it invented one). Both mean the toolbox went unused; only one of them is a
    hallucination.
    """

    name = "tool_selection_accuracy"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        trace = _trace(sample)
        if trace is None:
            return _no_trace(sample, self.name)

        if not trace.called_any:
            return MetricResult(
                sample_id=sample.sample_id,
                metric_name=self.name,
                value=False,
                passed=False,
                details={"verdict": "called no tool — did the arithmetic in its head"},
            )

        hallucinated = trace.hallucinated_tools
        real = [c for c in trace.calls if c.tool_exists]
        ok = bool(real)
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=ok,
            passed=ok,
            details={
                "tools_called": [c.tool_name for c in trace.calls],
                "hallucinated": list(hallucinated),
                "verdict": (
                    "selected a real tool"
                    if ok
                    else f"INVENTED tools that do not exist: {list(hallucinated)}"
                ),
            },
        )


@register_metric("tool_execution_success")
class ToolExecutionSuccess(Metric):
    """Of the tools it called, did any actually run?

    Separates a **schema** failure (malformed arguments — the model's JSON was wrong) from a **plan**
    failure (valid call, but the tool said no: no such column, division by zero). These are not the
    same problem and they do not have the same fix.
    """

    name = "tool_execution_success"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        trace = _trace(sample)
        if trace is None:
            return _no_trace(sample, self.name)
        if not trace.called_any:
            return _not_applicable(sample, self.name, "no tool was called — nothing to execute")

        executed = [c for c in trace.calls if c.executed]
        ok = bool(executed)
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=ok,
            passed=ok,
            details={
                "n_calls": len(trace.calls),
                "n_executed": len(executed),
                "invalid_arguments": [c.tool_name for c in trace.calls if not c.arguments_valid],
                "errors": [c.error[:80] for c in trace.calls if c.error][:5],
            },
        )


def _not_applicable(sample: CanonicalSample, metric: str, reason: str) -> MetricResult:
    return MetricResult(
        sample_id=sample.sample_id,
        metric_name=metric,
        value=None,
        passed=None,
        details={"reason": reason},
    )


@register_metric("tool_result_utilization")
class ToolResultUtilization(Metric):
    """**The one that matters.** The tool worked — did the model then use what it said?

    A model that calls the calculator, receives ``40.55``, and writes "approximately 38" has not made
    an arithmetic error. It made a *trust* error, and every end-to-end metric in this platform will
    misattribute it: accuracy sees a wrong number and blames the sums.

    Only applicable when a tool actually executed. If nothing ran, there was no result to ignore, and
    scoring that as "ignored the result" would blame the model for the tool's failure.
    """

    name = "tool_result_utilization"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        trace = _trace(sample)
        if trace is None:
            return _no_trace(sample, self.name)
        if not trace.executed_any:
            return _not_applicable(
                sample, self.name, "no tool executed — there was no result to use or ignore"
            )

        used = trace.result_used
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=used,
            passed=used,
            details={
                "final_answer": trace.final_answer[:120],
                "tool_outputs": [c.output[:60] for c in trace.calls if c.executed][:4],
                "verdict": (
                    "used the tool's answer"
                    if used
                    else "IGNORED its own tool — called it, got a number, and wrote a different one"
                ),
            },
        )


@register_metric("tool_security_rejection")
class ToolSecurityRejection(Metric):
    """Did the model try to make the sandbox execute something that is not arithmetic?

    ``passed=True`` means it never tried.

    This is not scored to punish the model — a small model emitting ``__import__`` is confused, not
    malicious. It is scored because **the sandbox's refusal is the thing under test**. A run where a
    model probed the boundary and the boundary held is evidence the boundary works; a run where
    nothing was ever probed is no evidence at all, and says so.
    """

    name = "tool_security_rejection"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        trace = _trace(sample)
        if trace is None:
            return _no_trace(sample, self.name)

        refusals = trace.security_refusals
        clean = refusals == 0
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=clean,
            passed=clean,
            details={
                "sandbox_refusals": refusals,
                "verdict": (
                    "never asked the sandbox to run anything but arithmetic"
                    if clean
                    else f"the sandbox refused {refusals} non-arithmetic expression(s) — and held"
                ),
            },
        )
