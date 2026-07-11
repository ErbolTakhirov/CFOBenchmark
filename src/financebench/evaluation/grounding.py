"""Grounding, evidence, and hallucination metrics — **ours, not any benchmark's**.

FinanceBench ships no evaluator, so nothing here can claim to be official, and none of these
numbers may be compared with a published FinanceBench figure. They are named `financebench_*` so
that confusion is impossible.

The most important metric in this module is :class:`UnsupportedNumericClaim`, and it is worth
saying why.

A financial model's dangerous failure is not "wrong". It is **confidently wrong with a plausible
number** — a figure that looks like it came from the filing and did not. Answer accuracy cannot see
this: a model that hallucinates $1,577M when the answer is $1,577M scores identically to one that
read it off the page. So the check is *provenance*, not correctness: **every number the model
states must appear in the evidence it was given.** One that doesn't is an invented figure, and it
is a failure whether or not the final answer happened to be right.

This is fully deterministic, needs no judge, and applies to all 150 questions regardless of whether
the gold answer is a number, a yes/no, or an essay.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from financebench.evaluation.metrics.base import Metric, register_metric
from financebench.evaluation.numeric import parse_numeric_answer
from financebench.schemas.metric import MetricResult
from financebench.schemas.prediction import Prediction
from financebench.schemas.sample import CanonicalSample

__all__ = [
    "FinanceBenchAnswerAccuracy",
    "FinanceBenchCitationAccuracy",
    "UnsupportedNumericClaim",
    "number_is_supported",
    "numbers_in",
]

#: Any number-ish token: 1,577  ·  1577.00  ·  (1,577)  ·  -1577  ·  1.5%
_NUMBER_RE = re.compile(r"\(?-?\$?\s*\d[\d,]*\.?\d*\s*\)?%?")

#: Numbers too small or too common to be evidence of anything. A model saying "the two segments" or
#: "increased by 1" is not making a financial claim, and flagging it would drown the real signal.
_TRIVIAL = frozenset({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0})

#: How close a stated number must be to one in the evidence to count as "the same number".
#: Deliberately loose: a model that rounds $1,577.4M to $1,577M has not hallucinated. A model that
#: says $1,600M has.
_RELATIVE_TOLERANCE = 0.005


def numbers_in(text: str) -> list[float]:
    """Every number a piece of text states, normalized."""
    out: list[float] = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group().strip()
        if not any(ch.isdigit() for ch in token):
            continue
        parsed = parse_numeric_answer(token)
        if parsed is None or parsed.resolved_value is None:
            continue
        out.append(parsed.resolved_value)
    return out


def number_is_supported(value: float, evidence_numbers: Iterable[float]) -> bool:
    """Does ``value`` appear in the evidence — as itself, at a reporting scale of it, or as its
    magnitude?

    Two allowances, and both are necessary or the metric is useless:

    **Scale.** A filing reports "1,577" in a table headed *(millions)*, and the model answers
    "$1,577 million". Same fact. Without this, every correctly-scaled answer is flagged as invented.

    **Sign.** A cash-flow statement writes capital expenditure as ``(1,577)`` — a parenthesised
    *outflow*, i.e. minus 1577 - while FinanceBench's gold answer for the same figure is ``$1577.00``,
    positive. They are the same number in the filing; the sign is an accounting convention about
    direction, not a different fact. Comparing signed values marked the true figure as a
    hallucination, which is precisely backwards.

    (A wrong *sign* in an answer is still caught — by the wrong-sign failure attribution, which is
    where it belongs. This function asks only "did this figure come from the document?")
    """
    if abs(value) in _TRIVIAL:
        return True
    for candidate in evidence_numbers:
        for factor in (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9, 0.01, 100.0):
            scaled = abs(candidate) * factor
            if scaled == 0:
                continue
            if abs(abs(value) - scaled) / max(scaled, 1e-9) <= _RELATIVE_TOLERANCE:
                return True
    return False


def _evidence_text(sample: CanonicalSample) -> str:
    """Everything the model was actually shown, plus the gold evidence snippets.

    Both, deliberately. In `context_given` these are the same thing. In `retrieval_required` the
    model saw only what the retriever found — and a number it took from a *retrieved* page is not
    invented even if that page was the wrong one. Wrong-page is a *retrieval* failure and is
    attributed as one; it must not also be counted as a hallucination, or the two failure modes
    become impossible to tell apart, and they have opposite fixes.
    """
    parts = list(sample.context.text)
    parts.extend(
        " | ".join(cell for cell in row) for table in sample.context.tables for row in table.rows
    )
    parts.extend(e.text_snippet or "" for e in sample.gold.evidence)
    return "\n".join(parts)


@register_metric("financebench_unsupported_numeric_claim")
class UnsupportedNumericClaim(Metric):
    """Did the model state a number that appears nowhere in its evidence?

    **A hallucination detector, not an accuracy metric.** It fires even when the final answer is
    right, because a model that guesses correctly is still guessing. ``passed=True`` means *no*
    unsupported claim was made — so a passing score here is a good thing.
    """

    name = "financebench_unsupported_numeric_claim"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        response = prediction.response
        if response is None or response.financial_answer is None:
            return MetricResult(
                sample_id=sample.sample_id,
                metric_name=self.name,
                value=False,
                passed=False,
                details={"reason": "no answer"},
            )
        answer = response.financial_answer

        # A refusal states no numbers, so it can invent none. It may be a bad refusal — that is the
        # calibration metric's business, not this one's.
        if answer.insufficient_information:
            return MetricResult(
                sample_id=sample.sample_id,
                metric_name=self.name,
                value=True,
                passed=True,
                details={"reason": "declined to answer — no numeric claim made"},
            )

        stated = numbers_in(f"{answer.answer} {answer.brief_explanation or ''}")
        if answer.numeric_value is not None:
            stated.append(answer.numeric_value)

        evidence_numbers = numbers_in(_evidence_text(sample))
        unsupported = [n for n in stated if not number_is_supported(n, evidence_numbers)]

        clean = not unsupported
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=clean,
            passed=clean,
            details={
                "n_numbers_stated": len(stated),
                "n_unsupported": len(unsupported),
                "unsupported": unsupported[:8],
                "n_evidence_numbers": len(evidence_numbers),
            },
        )


@register_metric("financebench_answer_accuracy")
class FinanceBenchAnswerAccuracy(Metric):
    """Answer correctness — **only where it is deterministically checkable.**

    52 of the 150 gold answers are numbers and 37 are yes/no. Those are scored here. The other 61
    are multi-sentence analyses, and exact-matching an essay is not evaluation, it is theatre. They
    return ``passed=None`` — *not applicable*, not zero — and are picked up by the optional judge.

    Reporting a fabricated 0.0 for an answer nobody could check is exactly the kind of number this
    project exists to refuse.
    """

    name = "financebench_answer_accuracy"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        shape = sample.metadata.get("answer_shape", "analytical")

        if shape == "analytical":
            return MetricResult(
                sample_id=sample.sample_id,
                metric_name=self.name,
                value=None,
                passed=None,  # not applicable — NOT a failure
                details={
                    "reason": "gold answer is a free-text analysis; not deterministically "
                    "checkable. Scored by the judge, if one is configured.",
                    "answer_shape": shape,
                },
            )

        response = prediction.response
        if response is None or response.financial_answer is None:
            return MetricResult(
                sample_id=sample.sample_id,
                metric_name=self.name,
                value=False,
                passed=False,
                details={"reason": "no answer"},
            )
        answer = response.financial_answer

        if shape == "boolean":
            gold_yes = sample.gold.answer.strip().lower().startswith("yes")
            said = f"{answer.answer} {answer.brief_explanation or ''}".strip().lower()
            match = re.match(r"^\s*(yes|no)\b", said)
            if match is None:
                return MetricResult(
                    sample_id=sample.sample_id,
                    metric_name=self.name,
                    value=False,
                    passed=False,
                    details={"reason": "gold is yes/no but the answer commits to neither"},
                )
            predicted_yes = match.group(1) == "yes"
            correct = predicted_yes == gold_yes
            return MetricResult(
                sample_id=sample.sample_id,
                metric_name=self.name,
                value=correct,
                passed=correct,
                details={"predicted": match.group(1), "gold": "yes" if gold_yes else "no"},
            )

        # numeric
        gold = sample.gold.numeric_value
        predicted = answer.numeric_value
        if predicted is None:
            parsed = parse_numeric_answer(answer.answer)
            predicted = parsed.resolved_value if parsed is not None else None
        if gold is None or predicted is None:
            return MetricResult(
                sample_id=sample.sample_id,
                metric_name=self.name,
                value=False,
                passed=False,
                details={"reason": "no extractable number", "gold": gold, "predicted": predicted},
            )

        # 1 % relative: filings round, and so do analysts. A 1 % band forgives rounding without
        # forgiving a wrong figure.
        correct = abs(predicted - gold) <= max(abs(gold) * 0.01, 1e-6) or number_is_supported(
            predicted, [gold]
        )
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=correct,
            passed=correct,
            details={"predicted": predicted, "gold": gold},
        )


@register_metric("financebench_citation_accuracy")
class FinanceBenchCitationAccuracy(Metric):
    """Did the model cite the document the answer actually came from?

    Only meaningful when the model was asked for citations (`grounded_citations_v1`). A model that
    was never asked to cite is not failing to cite, and returns *not applicable* rather than zero.
    """

    name = "financebench_citation_accuracy"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        response = prediction.response
        if response is None or response.financial_answer is None:
            return MetricResult(
                sample_id=sample.sample_id,
                metric_name=self.name,
                value=False,
                passed=False,
                details={"reason": "no answer"},
            )
        citations = response.financial_answer.citations
        if not citations:
            return MetricResult(
                sample_id=sample.sample_id,
                metric_name=self.name,
                value=None,
                passed=None,
                details={"reason": "no citation offered (was the model asked for one?)"},
            )

        gold_docs = {e.document_id for e in sample.gold.evidence if e.document_id}
        cited = {c.document_id for c in citations if c.document_id}
        hit = bool(gold_docs & cited)
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=hit,
            passed=hit,
            details={"cited": sorted(cited), "gold_documents": sorted(gold_docs)},
        )
