"""TAT-QA's native evaluator, ported faithfully from ``tatqa_metric.py`` / ``tatqa_utils.py``.

TAT-QA reports **exact match**, a **numeracy-aware F1**, and a separate **scale accuracy**. Three
things about the official implementation are easy to get wrong, and all three change the numbers:

1. **The F1 uses a Hungarian assignment.** ``tatqa_metric.py`` really does
   ``from scipy.optimize import linear_sum_assignment`` — for a multi-span answer it finds the
   optimal 1-to-1 matching between predicted and gold spans before averaging their token-F1s. A
   naive bag-overlap would disagree, so scipy is a genuine dependency, not a convenience.
2. **``f1 := em`` is forced** when the gold answer type is ``arithmetic`` or ``count``. A number is
   either right or it isn't; partial token credit for a wrong number would be nonsense.
3. **Scale is folded into the compared string**, not scored inside EM/F1:
   ``'%.4f' % (round(n, 2) * scale_to_num(scale))``, with ``percent -> 0.01``. So a model that says
   "23.42" with scale ``percent`` and one that says "0.2342" with no scale must both match a gold of
   23.42/percent — which is what ``add_percent_pred`` is for.

**One part is ours, and is labelled as such.** The official evaluator is handed a *pair*
``[answer, scale]``, because TAT-QA's reference model (TagOp) has a dedicated scale-classification
head. A general LLM has no such head — it just writes prose. So :func:`extract_answer_and_scale`
infers the scale from the model's structured answer, and that inference is *not* part of the
official metric. The parity suite therefore feeds ``(answer, scale)`` pairs straight into both
implementations, bypassing extraction, to prove the *metric* is official; the extraction is tested
separately on its own terms.
"""

from __future__ import annotations

import math
import re
import string
from collections.abc import Sequence

from scipy.optimize import linear_sum_assignment

from financebench.evaluation.metrics.base import Metric, register_metric
from financebench.schemas.metric import MetricResult
from financebench.schemas.prediction import Prediction
from financebench.schemas.sample import CanonicalSample

__all__ = [
    "TatQAExactMatch",
    "TatQAF1",
    "TatQAScaleAccuracy",
    "extract_answer_and_scale",
    "get_answer_str",
    "get_metrics",
    "normalize_answer",
    "scale_to_num",
    "tatqa_em_and_f1",
    "to_number",
]

SCALES = ("", "thousand", "million", "billion", "percent")


# --------------------------------------------------------------------------- tatqa_utils.py


def scale_to_num(scale: str) -> int | float:
    """Port of the official ``scale_to_num``.

    The int/float distinction here is **load-bearing, not sloppiness**. The official function
    returns Python ``int`` for every scale except percent, and that type flows through ``to_number``
    into ``round()`` — and ``round(-134, 4)`` is the int ``-134`` while ``round(-134.0, 4)`` is the
    float ``-134.0``. ``normalize_number`` then ``str()``s the result, so returning 1.0 instead of 1
    turns the normalized answer "-134" into "-134.0", which no longer exact-matches a gold of
    "-134". Caught by tests/parity/test_tatqa_parity.py; do not "tidy" these into floats.
    """
    scale = scale.lower()
    if "hundred" in scale:
        return 100
    if "thousand" in scale:
        return 1000
    if "million" in scale:
        return 1_000_000
    if "billion" in scale:
        return 1_000_000_000
    if "percent" in scale:
        return 0.01
    return 1


_EXCLUDE_IN_NUM = "'\"\\$€£¥%(),[]"


def _clean_num(text: str) -> str:
    return "".join(ch for ch in str(text) if ch not in _EXCLUDE_IN_NUM)


def extract_one_num_from_str(text: str) -> float | int | None:
    cleaned = _clean_num(text)
    groups = re.findall(r"([+-]?\d+(\.\d+)?)|([+-]?\.\d+)", cleaned)
    if not groups:
        return None
    num = groups[0][0]
    if num == "":
        return None
    return float(num) if "." in num else int(num)


def is_number(text: str) -> bool:
    try:
        words = " ".join(_clean_num(word) for word in text.split()).split()
        if not words:
            return False
        num = float(words[0])
        if math.isnan(num):
            return False
        # A trailing word that isn't a scale word ("5 apples") means this isn't a number.
        return not (len(words) >= 2 and scale_to_num(words[1]) == 1)
    except ValueError:
        return False


def _negative_num_handle(text: str) -> int:
    """``(134)`` means ``-134`` in a financial table."""
    return -1 if re.findall(r"(\([\d.\s]+\))", text.strip()) else 1


def _percent_num_handle(text: str) -> int | float:
    # int 1, not 1.0 — see scale_to_num's docstring.
    return 0.01 if re.findall(r"([\d.\s]+%)", text.strip()) else 1


def _word_scale_handle(text: str) -> int | float:
    for match in re.finditer(r"([\d.]+\s?[a-zA-Z]+)", text):
        return scale_to_num(match.group(0).lower())
    return 1


def to_number(text: str) -> int | float | None:
    """Port of the official ``to_number``. Returns an ``int`` when every factor is an int — which
    matters, because the result gets ``str()``d into the normalized answer. See ``scale_to_num``."""
    num = extract_one_num_from_str(text)
    if num is None:
        return None
    return round(
        num * _word_scale_handle(text) * _negative_num_handle(text) * _percent_num_handle(text), 4
    )


def _remove_articles(text: str) -> str:
    return re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


_PUNCTUATION = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not is_number(text):
        return "".join(ch for ch in text if ch not in _PUNCTUATION)
    return text


def _normalize_number(text: str) -> str:
    return str(to_number(text)) if is_number(text) else text


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation/articles, normalize numbers, squash whitespace."""
    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(token.lower()))))
        for token in re.split(" ", text)
    ]
    return " ".join(part for part in parts if part.strip()).strip()


# --------------------------------------------------------------------------- tatqa_metric.py


def _answer_to_bags(answer: str | Sequence[str]) -> tuple[list[str], list[set[str]]]:
    raw_spans = [answer] if isinstance(answer, str) else list(answer)
    normalized: list[str] = []
    bags: list[set[str]] = []
    for raw in raw_spans:
        span = normalize_answer(str(raw))
        normalized.append(span)
        bags.append(set(span.split()))
    return normalized, bags


def _compute_f1(predicted_bag: set[str], gold_bag: set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    precision = 1.0 if not predicted_bag else intersection / float(len(predicted_bag))
    recall = 1.0 if not gold_bag else intersection / float(len(gold_bag))
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _align_bags(predicted: list[set[str]], gold: list[set[str]]) -> list[float]:
    """The Hungarian assignment: the optimal 1-to-1 matching of predicted spans to gold spans.

    This is the official behaviour (``tatqa_metric.py`` imports ``scipy.optimize.
    linear_sum_assignment``), not an embellishment — a naive bag-overlap gives different numbers on
    multi-span answers. The cost matrix is a plain list-of-lists; scipy converts it, so numpy need
    not be imported here.
    """
    scores = [[_compute_f1(pred, gold_item) for pred in predicted] for gold_item in gold]
    negated = [[-value for value in row] for row in scores]
    row_ind, col_ind = linear_sum_assignment(negated)

    max_scores = [0.0] * max(len(gold), len(predicted))
    for row, column in zip(row_ind, col_ind, strict=True):
        max_scores[row] = max(max_scores[row], scores[row][column])
    return max_scores


def get_metrics(predicted: str | Sequence[str], gold: str | Sequence[str]) -> tuple[float, float]:
    """Official ``(exact_match, f1)`` for one predicted/gold answer pair."""
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    exact_match = (
        1.0
        if set(predicted_bags[0]) == set(gold_bags[0])
        and len(predicted_bags[0]) == len(gold_bags[0])
        else 0.0
    )
    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    mean_f1 = sum(f1_per_bag) / len(f1_per_bag) if f1_per_bag else 0.0
    return exact_match, round(mean_f1, 2)


def get_answer_str(answers: Sequence[object], scale: str) -> list[str]:
    """Fold the scale into the answer string, exactly as the official evaluator does."""
    out: list[str] = []
    for answer in sorted(answers, key=str):
        text = str(answer)
        if is_number(text):
            number = to_number(text)
            if number is None:
                if scale:
                    text = f"{text} {scale}"
            elif "%" in text:  # the answer is already a percentage
                text = "%.4f" % number
            else:
                text = "%.4f" % (round(number, 2) * scale_to_num(scale))
        elif scale:
            text = f"{text} {scale}"
        out.append(text)
    return [" ".join(out)]


def _add_percent_pred(
    prediction_strings: list[str], pred_scale: str, prediction: Sequence[object]
) -> list[str]:
    """Resolves 0.2342-vs-23.42%: offer the bare-number reading as an alternative."""
    if len(prediction) > 1:
        return prediction_strings
    text = str(prediction[0])
    if not pred_scale and "%" not in text and is_number(text):
        number = to_number(text)
        if number is not None:
            prediction_strings.append("%.4f" % number)
    return prediction_strings


def tatqa_em_and_f1(
    predicted: Sequence[object],
    pred_scale: str,
    gold: Sequence[object],
    gold_scale: str,
    gold_answer_type: str,
) -> tuple[float, float]:
    """The official per-sample EM/F1, including the two rules that catch people out."""
    if not predicted:
        return 0.0, 0.0

    gold_strings = get_answer_str(gold, gold_scale)
    pred_strings = _add_percent_pred(get_answer_str(predicted, pred_scale), pred_scale, predicted)

    best = (0.0, 0.0)
    for pred in pred_strings:
        for gold_string in gold_strings:
            best = max(best, get_metrics(pred, gold_string))
    exact_match, f1 = best

    # A number is right or it isn't; partial token credit for a wrong number is nonsense.
    if gold_answer_type in ("arithmetic", "count"):
        f1 = exact_match
    return exact_match, f1


# --------------------------------------------------------------------------- OURS: extraction

_SCALE_WORDS: dict[str, str] = {
    "thousand": "thousand",
    "thousands": "thousand",
    "k": "thousand",
    "million": "million",
    "millions": "million",
    "m": "million",
    "mn": "million",
    "billion": "billion",
    "billions": "billion",
    "bn": "billion",
    "b": "billion",
    "percent": "percent",
    "percentage": "percent",
    "%": "percent",
    "pct": "percent",
}


def extract_answer_and_scale(prediction: Prediction) -> tuple[list[str], str]:
    """Turn a model's structured answer into the ``(answer, scale)`` pair the official evaluator
    expects.

    **Not part of the official metric.** TAT-QA's reference model has a dedicated scale-prediction
    head; an LLM does not, so the scale has to be read back out of what it wrote. Conservative on
    purpose: an unrecognised unit yields ``""`` (no scale) rather than a guess, because a wrong
    scale turns a correct number into a wrong one by a factor of a thousand.
    """
    response = prediction.response
    if response is None or response.financial_answer is None:
        return [], ""
    answer = response.financial_answer

    unit = (answer.unit or "").strip().lower()
    scale = _SCALE_WORDS.get(unit, "")
    if not scale:
        # Fall back to the answer text: "23.4%" or "$1.2 million".
        text = answer.answer.lower()
        if "%" in text:
            scale = "percent"
        else:
            for word, mapped in _SCALE_WORDS.items():
                if len(word) > 2 and re.search(rf"\b{re.escape(word)}\b", text):
                    scale = mapped
                    break

    # Multi-span answers: the model is asked for a JSON string, so split on the obvious separators.
    raw = answer.answer.strip()
    if answer.numeric_value is not None and not raw:
        return [str(answer.numeric_value)], scale
    spans = [part.strip() for part in re.split(r"\s*[;,]\s*(?![\d])|\s*\|\s*", raw) if part.strip()]
    return (spans or [raw]), scale


# --------------------------------------------------------------------------- metrics


def _gold(sample: CanonicalSample) -> tuple[list[str], str, str]:
    gold_answers = list(sample.gold.acceptable_answers) or [sample.gold.answer]
    gold_scale = sample.metadata.get("scale", "")
    answer_type = sample.metadata.get("answer_type", "span")
    return gold_answers, gold_scale, answer_type


@register_metric("tatqa_exact_match")
class TatQAExactMatch(Metric):
    """TAT-QA's **official** exact match."""

    name = "tatqa_exact_match"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        predicted, pred_scale = extract_answer_and_scale(prediction)
        gold_answers, gold_scale, answer_type = _gold(sample)
        em, _ = tatqa_em_and_f1(predicted, pred_scale, gold_answers, gold_scale, answer_type)
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=em,
            passed=em == 1.0,
            details={
                "predicted": predicted,
                "pred_scale": pred_scale,
                "gold": gold_answers,
                "gold_scale": gold_scale,
                "answer_type": answer_type,
            },
        )


@register_metric("tatqa_f1")
class TatQAF1(Metric):
    """TAT-QA's **official** numeracy-aware F1 (Hungarian-aligned token-bag F1)."""

    name = "tatqa_f1"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        predicted, pred_scale = extract_answer_and_scale(prediction)
        gold_answers, gold_scale, answer_type = _gold(sample)
        _, f1 = tatqa_em_and_f1(predicted, pred_scale, gold_answers, gold_scale, answer_type)
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=f1,
            passed=f1 == 1.0,
            details={"predicted": predicted, "gold": gold_answers, "f1": f1},
        )


@register_metric("tatqa_scale_accuracy")
class TatQAScaleAccuracy(Metric):
    """Did the model get the *magnitude* right?

    Tracked on its own because a right number at the wrong scale is not a near-miss — it is off by
    a factor of a thousand, and in a financial context that is the difference between a rounding
    error and a catastrophe. It feeds the ``wrong_scale_rate`` critical gate.
    """

    name = "tatqa_scale_accuracy"

    def score(self, sample: CanonicalSample, prediction: Prediction) -> MetricResult:
        _, pred_scale = extract_answer_and_scale(prediction)
        _, gold_scale, _ = _gold(sample)
        correct = pred_scale == gold_scale
        return MetricResult(
            sample_id=sample.sample_id,
            metric_name=self.name,
            value=correct,
            passed=correct,
            details={"pred_scale": pred_scale, "gold_scale": gold_scale},
        )
