"""ConvFinQA metrics — per turn. The conversation-level story is in ``evaluation/conversation.py``.

ConvFinQA's official evaluator grades a *turn* two ways, and this module reuses FinQA's
parity-tested machinery for both, because ConvFinQA's programs are FinQA's programs:

- **execution accuracy** — run the predicted program, compare its result to the gold answer.
- **program accuracy** — symbolic equivalence against the gold program (``equal_program``).

Both are only defined when the run actually asked the model for a program. A direct-answer run gets
:class:`ConvFinQATurnAccuracy` instead — ours, tolerance-based, and named so it can never be read as
the official number.

The distinction that matters here is *between* turns, not within them: a turn is graded identically
under both conversation protocols. What changes is what the model was shown before it answered, and
that is the whole experiment. See ``schemas/common.ConversationProtocol``.
"""

from __future__ import annotations

from financebench.evaluation.metrics.base import Metric, register_metric
from financebench.evaluation.native.finqa import (
    equal_program,
    eval_program,
    extract_program,
    program_tokenization,
)
from financebench.evaluation.numeric import numeric_match, parse_numeric_answer
from financebench.prompts.profiles import create_prompt_profile
from financebench.schemas.metric import MetricResult
from financebench.schemas.prediction import Prediction
from financebench.schemas.sample import CanonicalSample

__all__ = [
    "ConvFinQAExecutionAccuracy",
    "ConvFinQAProgramAccuracy",
    "ConvFinQATurnAccuracy",
]

_TOLERANCE = 1e-3


def _asked_for_a_program(prediction: Prediction) -> bool:
    try:
        return create_prompt_profile(prediction.request.prompt_version).elicits_program
    except Exception:
        return False


def _fail(sample: CanonicalSample, metric: str, reason: str) -> MetricResult:
    return MetricResult(
        sample_id=sample.sample_id,
        metric_name=metric,
        value=False,
        passed=False,
        details={"reason": reason, "turn": sample.metadata.get("turn_index", "")},
    )


def _table(sample: CanonicalSample) -> list[list[str]]:
    if not sample.context.tables:
        return []
    return [list(row) for row in sample.context.tables[0].rows]


@register_metric("convfinqa_turn_accuracy")
class ConvFinQATurnAccuracy(Metric):
    """**Not ConvFinQA's metric.** Did the model state the right number (or yes/no) for this turn?

    ConvFinQA's official evaluator only ever grades programs, so for a direct-answer run there is no
    official number to report and this stands in — named so nobody mistakes it for one.
    """

    name = "convfinqa_turn_accuracy"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        response = prediction.response
        if response is None or response.financial_answer is None:
            return _fail(sample, self.name, "no parsed answer")
        answer = response.financial_answer

        # ConvFinQA's `greater` op yields yes/no rather than a number.
        if sample.gold.numeric_value is None:
            predicted = answer.answer.strip().casefold()
            gold = sample.gold.answer.strip().casefold()
            hit = predicted == gold
            return MetricResult(
                sample_id=sample.sample_id,
                metric_name=self.name,
                value=hit,
                passed=hit,
                details={"predicted": predicted, "gold": gold, "mode": "boolean"},
            )

        value = answer.numeric_value
        if value is None:
            parsed = parse_numeric_answer(answer.answer)
            value = parsed.resolved_value if parsed is not None else None
        if value is None:
            return _fail(sample, self.name, "predicted answer has no extractable number")

        gold_value = sample.gold.numeric_value
        hit = numeric_match(value, gold_value, absolute_tolerance=_TOLERANCE)

        # Same fraction-vs-percent convention as FinQA (they share a program language): the gold for
        # a percentage question is the raw division result, so a model that answers "16.4%" where
        # the gold is 0.164 is right, not wrong by a factor of 100.
        unit = (answer.unit or "").strip().lower()
        percent = unit in {"%", "percent", "percentage", "pct"} or "%" in (answer.answer or "")
        mode = "numeric"
        if (
            not hit
            and percent
            and numeric_match(value / 100.0, gold_value, absolute_tolerance=_TOLERANCE)
        ):
            hit, mode = True, "numeric_percent_reconciled"

        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=hit,
            passed=hit,
            details={
                "predicted": value,
                "gold": gold_value,
                "mode": mode,
                "turn": sample.metadata.get("turn_index", ""),
                "conversation": sample.metadata.get("conversation_id", ""),
            },
        )


@register_metric("convfinqa_execution_accuracy")
class ConvFinQAExecutionAccuracy(Metric):
    """ConvFinQA's **official** turn-level execution accuracy: execute the predicted program and
    compare its result to the gold answer.

    ConvFinQA's programs are FinQA's programs, so this is FinQA's parity-tested executor, not a
    re-implementation of it. Undefined — and therefore not scored — for a run that never asked the
    model for a program.
    """

    name = "convfinqa_execution_accuracy"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        if not _asked_for_a_program(prediction):
            return _fail(
                sample,
                self.name,
                "run did not use a program-eliciting prompt profile; official execution accuracy "
                "is undefined for a free-text answer",
            )
        response = prediction.response
        if response is None:
            return _fail(sample, self.name, "no response")

        program = extract_program(response.content)
        if program is None:
            return _fail(sample, self.name, "no program found in model output")

        invalid, result = eval_program(program_tokenization(program), _table(sample))
        gold = (
            sample.gold.numeric_value
            if sample.gold.numeric_value is not None
            else sample.gold.answer.strip().casefold()
        )
        hit = invalid == 0 and result == gold
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=hit,
            passed=hit,
            details={
                "predicted_program": program,
                "executed": result,
                "gold": gold,
                "invalid_program": bool(invalid),
                "turn": sample.metadata.get("turn_index", ""),
            },
        )


@register_metric("convfinqa_program_accuracy")
class ConvFinQAProgramAccuracy(Metric):
    """ConvFinQA's **official** turn-level program accuracy: symbolic equivalence with the gold
    program.

    A model can reach the right number by the wrong route — ``(a - b) / b`` and ``a / b - 1`` agree
    on the answer and disagree on the reasoning. Execution accuracy cannot see the difference;
    this can.
    """

    name = "convfinqa_program_accuracy"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        if not _asked_for_a_program(prediction):
            return _fail(
                sample,
                self.name,
                "run did not use a program-eliciting prompt profile; there is no program to score",
            )
        response = prediction.response
        if response is None:
            return _fail(sample, self.name, "no response")
        if not sample.gold.program:
            return _fail(sample, self.name, "turn has no gold program")

        program = extract_program(response.content)
        if program is None:
            return _fail(sample, self.name, "no program found in model output")

        hit = equal_program(
            program_tokenization(sample.gold.program), program_tokenization(program)
        )
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=hit,
            passed=hit,
            details={
                "predicted_program": program,
                "gold_program": sample.gold.program,
                "turn": sample.metadata.get("turn_index", ""),
            },
        )
