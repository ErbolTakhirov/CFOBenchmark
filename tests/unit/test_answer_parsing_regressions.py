"""Regressions from a live model run — every one of these silently deflated a real score.

These were not found by unit tests. They were found by running qwen2.5:3b against 40 real FinQA
questions, getting 5 %, and refusing to believe it before checking whether the pipeline was at
fault. It was: the benchmark was measuring its own plumbing. Fixing these took the *same cached
responses* from 5 % to 15 %.

Every case below is a thing a real model actually wrote.
"""

from __future__ import annotations

import pytest

from financebench.schemas.model_io import FinancialAnswer, ModelResponse


def test_a_null_boolean_does_not_destroy_the_whole_answer() -> None:
    """THE bug. qwen2.5:3b routinely writes `"insufficient_information": null`.

    A strict `bool` rejects that; `model_validate` fails; `from_text` falls back to treating the
    entire raw JSON blob as the answer *string* — and throws away the perfectly good
    `"numeric_value": 29.3` sitting right next to it. Roughly half of all valid answers were being
    discarded this way, and the model was being blamed for it.
    """
    raw = (
        '{"answer": "29.3%", "numeric_value": 29.3, "unit": "%", "period": null, '
        '"citations": [], "insufficient_information": null, "confidence": 1.0}'
    )
    answer = FinancialAnswer.from_text(raw)

    assert answer is not None
    assert answer.numeric_value == 29.3, "the number was right there in the response"
    assert answer.answer == "29.3%"
    assert answer.insufficient_information is False, "null means 'not flagged', not 'invalid'"
    assert not answer.answer.startswith("{"), "the raw JSON blob must not become the answer"


def test_a_null_confidence_is_also_survivable() -> None:
    answer = FinancialAnswer.from_text(
        '{"answer": "42", "numeric_value": 42, "confidence": null, '
        '"insufficient_information": null}'
    )
    assert answer is not None
    assert answer.numeric_value == 42


def test_a_bare_number_in_the_answer_field_is_an_answer_not_a_type_error() -> None:
    answer = FinancialAnswer.from_text('{"answer": 42, "numeric_value": 42}')
    assert answer is not None
    assert answer.answer == "42"
    assert answer.numeric_value == 42


def test_the_period_field_the_prompt_asks_for_actually_exists() -> None:
    """The prompt's JSON schema advertises `period`. If the model fills it and the envelope has no
    such field, `extra="ignore"` silently drops it — and a wrong-period failure becomes
    undetectable."""
    answer = FinancialAnswer.from_text('{"answer": "42", "numeric_value": 42, "period": "FY2018"}')
    assert answer is not None
    assert answer.period == "FY2018"


@pytest.mark.parametrize(
    "raw",
    [
        '{"answer": "10%", "numeric_value": null, "unit": "%"}',
        '```json\n{"answer": "10%", "numeric_value": null}\n```',
        'Here is the answer:\n{"answer": "10%", "numeric_value": null}',
    ],
)
def test_a_missing_numeric_value_still_leaves_a_parseable_answer_string(raw: str) -> None:
    """When the model leaves `numeric_value` null but writes the number in `answer`, the metric's
    fallback parse must have something to work with — which means `answer` must be "10%", not the
    raw blob."""
    answer = FinancialAnswer.from_text(raw)
    assert answer is not None
    assert answer.answer == "10%"


def test_a_response_that_is_genuinely_unparseable_is_still_recorded_not_dropped() -> None:
    answer = FinancialAnswer.from_text("I'm afraid I can't help with that.")
    assert answer is not None
    assert answer.answer == "I'm afraid I can't help with that."
    assert answer.numeric_value is None


def test_an_empty_response_yields_nothing_rather_than_a_fabricated_answer() -> None:
    assert FinancialAnswer.from_text("") is None
    assert FinancialAnswer.from_text("   ") is None


def test_the_parse_is_derived_from_content_not_frozen_into_the_cache() -> None:
    """The second-order bug: the cache stored the *parse* alongside the content, so fixing the
    parser changed nothing for any sample that had already been run — and re-scoring kept
    reporting the old, broken parse. The engine now re-derives the answer from the raw content on
    every cache read.

    Content is ground truth. The parse is our code's opinion about it, and our code keeps
    improving.
    """
    from financebench.execution.engine import _reparse

    raw = '{"answer": "29.3%", "numeric_value": 29.3, "insufficient_information": null}'
    # A response as an OLD, buggy parser would have cached it: the blob dumped into `answer`.
    stale = ModelResponse(
        provider="ollama",
        model="qwen2.5:3b",
        content=raw,
        financial_answer=FinancialAnswer(answer=raw),  # the broken fallback
        parsed=True,
    )
    assert stale.financial_answer is not None
    assert stale.financial_answer.numeric_value is None  # the old, wrong parse

    fresh = _reparse(stale)
    assert fresh.financial_answer is not None
    assert fresh.financial_answer.numeric_value == 29.3, "today's parser must win"
    assert fresh.content == raw, "the raw content is never rewritten"
