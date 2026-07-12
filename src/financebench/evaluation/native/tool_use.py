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


@register_metric("tool_invocation_rate")
class ToolInvocationRate(Metric):
    """Did the model call ANY tool — real or invented?

    Distinct from ``tool_selection_accuracy``, which conflates "called nothing" with "called
    something that does not exist". Those have opposite fixes: a model that never reaches for the
    toolbox needs a better prompt, and a model that invents `compute_ratio()` needs a better schema.
    A single number that says "0.167" cannot tell you which one you have.
    """

    name = "tool_invocation_rate"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        trace = _trace(sample)
        if trace is None:
            return _no_trace(sample, self.name)
        called = trace.called_any
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=called,
            passed=called,
            details={"n_calls": len(trace.calls), "turns": trace.turns},
        )


@register_metric("tool_argument_validity")
class ToolArgumentValidity(Metric):
    """Of the calls it made, how many had arguments the tool could actually accept?

    NOT APPLICABLE when it called nothing — a model that never picked up the calculator has not
    passed an argument-validity test, and scoring it 0.0 would blame it for a mistake it never made,
    while scoring it 1.0 would credit it for a competence it never showed.
    """

    name = "tool_argument_validity"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        trace = _trace(sample)
        if trace is None:
            return _no_trace(sample, self.name)
        real = [c for c in trace.calls if c.tool_exists]
        if not real:
            return _not_applicable(sample, self.name, "no real tool was ever called")
        valid = [c for c in real if c.arguments_valid]
        rate = len(valid) / len(real)
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=rate,
            passed=rate == 1.0,
            details={
                "n_calls": len(real),
                "n_valid": len(valid),
                "invalid": [
                    {"tool": c.tool_name, "args": c.arguments, "error": c.error}
                    for c in real
                    if not c.arguments_valid
                ],
            },
        )


@register_metric("tool_hallucination_rate")
class ToolHallucinationRate(Metric):
    """Did it invent a tool that does not exist?

    A model that calls ``compute_debt_ratio()`` when no such tool was offered is not making an
    arithmetic error — it is confabulating an API. In an agent wired to real systems, that call is
    either a crash or, worse, a silent no-op the model then narrates a result for.
    """

    name = "tool_hallucination_rate"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        trace = _trace(sample)
        if trace is None:
            return _no_trace(sample, self.name)
        if not trace.called_any:
            return _not_applicable(sample, self.name, "no tool was called, so none was invented")
        invented = list(trace.hallucinated_tools)
        clean = not invented
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=clean,
            passed=clean,
            details={"invented": invented, "offered": list(trace.tools_offered)},
        )


@register_metric("tool_error_recovery")
class ToolErrorRecovery(Metric):
    """After a tool told it "no", did it recover — or give up / repeat itself?

    The agent loop feeds every tool error back to the model, so a failed call is a chance to fix the
    arguments and try again. NOT APPLICABLE when nothing ever failed: a model that got everything
    right first time has demonstrated nothing about recovery, in either direction.
    """

    name = "tool_error_recovery"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        trace = _trace(sample)
        if trace is None:
            return _no_trace(sample, self.name)
        failed = [c for c in trace.calls if not c.executed]
        if not failed:
            return _not_applicable(sample, self.name, "no tool call ever failed")
        # It recovered if, after the first failure, some LATER call executed successfully.
        first_failure = min(c.turn for c in failed)
        recovered = any(c.executed and c.turn > first_failure for c in trace.calls)
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=recovered,
            passed=recovered,
            details={
                "n_failed_calls": len(failed),
                "first_failure_turn": first_failure,
                "verdict": (
                    "recovered after a tool error"
                    if recovered
                    else "never produced a working call after the first error"
                ),
            },
        )
