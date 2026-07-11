"""Tests for the TAT-QA glue that is **ours**, not official.

The metric itself (EM, numeracy F1, scale folding) is proven correct in ``tests/parity/`` by running
the real official evaluator side by side. What is *not* covered there — because it doesn't exist
upstream — is :func:`extract_answer_and_scale`: the official evaluator is handed a ready-made
``[answer, scale]`` pair, because TAT-QA's reference model has a scale-classification head. An LLM
just writes prose, so we have to read the scale back out of it.

That inference is the weak link, so it is tested on its own terms here. It is deliberately
conservative: an unrecognised unit yields no scale rather than a guess, because a wrong scale
doesn't make an answer slightly wrong — it makes it wrong by a factor of a thousand.
"""

from __future__ import annotations

import pytest
from tests.factories import make_sample

from financebench.evaluation.native.tatqa import (
    extract_answer_and_scale,
    scale_to_num,
    tatqa_em_and_f1,
    to_number,
)
from financebench.execution.engine import build_request
from financebench.schemas.model_io import FinancialAnswer, ModelResponse, ModelSpec
from financebench.schemas.prediction import Prediction
from financebench.schemas.run import RunConfig


def _prediction(answer: str, unit: str | None = None, numeric: float | None = None) -> Prediction:
    sample = make_sample(benchmark="tatqa", sample_id="tatqa:dev:1")
    request = build_request(sample, ModelSpec.parse("ollama/qwen2.5:7b"), RunConfig())
    return Prediction(
        sample_id=sample.sample_id,
        benchmark="tatqa",
        split="dev",
        request=request,
        created_at="t",
        response=ModelResponse(
            provider="ollama",
            model="qwen2.5:7b",
            content=answer,
            financial_answer=FinancialAnswer(answer=answer, unit=unit, numeric_value=numeric),
            parsed=True,
        ),
    )


# --------------------------------------------------------------------------- scale extraction


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("million", "million"),
        ("millions", "million"),
        ("m", "million"),
        ("thousand", "thousand"),
        ("billion", "billion"),
        ("percent", "percent"),
        ("%", "percent"),
        ("usd", ""),  # a currency is not a scale
        ("days", ""),
        (None, ""),
    ],
)
def test_scale_is_read_from_the_units_the_model_reports(unit: str | None, expected: str) -> None:
    _, scale = extract_answer_and_scale(_prediction("123", unit=unit))
    assert scale == expected


def test_scale_falls_back_to_the_answer_text_when_no_unit_is_given() -> None:
    _, scale = extract_answer_and_scale(_prediction("$1.2 million"))
    assert scale == "million"

    _, percent = extract_answer_and_scale(_prediction("up 23.4%"))
    assert percent == "percent"


def test_an_unrecognised_unit_yields_no_scale_rather_than_a_guess() -> None:
    """A wrong scale is not a near-miss — it is a 1000x error. Silence beats a guess."""
    _, scale = extract_answer_and_scale(_prediction("42", unit="widgets"))
    assert scale == ""


def test_a_multi_span_answer_is_split_into_its_spans() -> None:
    spans, _ = extract_answer_and_scale(
        _prediction("fixed-price type; cost-plus type; time-and-material type")
    )
    assert spans == ["fixed-price type", "cost-plus type", "time-and-material type"]


def test_a_decimal_number_is_not_split_on_its_comma_separator() -> None:
    """'1,496.5' is one number, not two spans. Splitting it would be catastrophic."""
    spans, _ = extract_answer_and_scale(_prediction("1,496.5"))
    assert spans == ["1,496.5"]


def test_a_prediction_with_no_response_extracts_to_nothing() -> None:
    sample = make_sample(benchmark="tatqa", sample_id="tatqa:dev:1")
    request = build_request(sample, ModelSpec.parse("ollama/qwen2.5:7b"), RunConfig())
    prediction = Prediction(
        sample_id=sample.sample_id,
        benchmark="tatqa",
        split="dev",
        request=request,
        created_at="t",
        response=None,
        error="boom",
    )
    spans, scale = extract_answer_and_scale(prediction)
    assert spans == []
    assert scale == ""


# --------------------------------------------------------------------------- the load-bearing
# int/float distinction (see scale_to_num's docstring)


def test_scale_to_num_returns_ints_not_floats_for_magnitudes() -> None:
    """Regression guard for a real parity bug: returning 1.0 instead of 1 makes
    ``normalize_answer("(134)")`` produce "-134.0" instead of "-134", which then fails to
    exact-match a gold of "-134". Caught by the parity suite."""
    assert scale_to_num("") == 1
    assert type(scale_to_num("")) is int
    assert type(scale_to_num("million")) is int
    assert type(scale_to_num("percent")) is float  # percent really is 0.01


def test_to_number_preserves_the_int_ness_of_a_whole_number() -> None:
    assert to_number("(134)") == -134
    assert type(to_number("(134)")) is int


# --------------------------------------------------------------------------- the official rules,
# spot-checked here too so a regression is visible without the reference clones


def test_f1_is_forced_to_em_for_arithmetic_so_a_near_miss_earns_nothing() -> None:
    em, f1 = tatqa_em_and_f1(["100"], "", ["100.5"], "", "arithmetic")
    assert (em, f1) == (0.0, 0.0), "partial token credit for a wrong number would be nonsense"


def test_a_right_number_at_the_wrong_scale_is_wrong() -> None:
    em, _ = tatqa_em_and_f1(["1496.5"], "thousand", ["1496.5"], "million", "arithmetic")
    assert em == 0.0
