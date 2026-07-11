"""Conversation-level analysis: full-conversation accuracy, first error, error propagation.

These are hand-built outcomes rather than model runs, because the point is to pin what the *formula*
says about a known situation. A model run would prove that the code executes; a fixture proves that
it computes the thing it claims to.
"""

from __future__ import annotations

import pytest

from financebench.evaluation.conversation import analyze_conversations
from financebench.schemas.common import AnswerType, ConversationProtocol, SplitOrigin
from financebench.schemas.metric import MetricResult
from financebench.schemas.sample import (
    CanonicalSample,
    EvaluationSpec,
    GoldAnswer,
    SampleContext,
    SourceInfo,
)

PROTOCOL = ConversationProtocol.MODEL_HISTORY


def _turn(conversation: str, index: int, *, reuses: tuple[int, ...] = ()) -> CanonicalSample:
    return CanonicalSample(
        benchmark="convfinqa",
        benchmark_version="test",
        split="dev",
        split_origin=SplitOrigin.OFFICIAL,
        sample_id=f"convfinqa:dev:{conversation}#t{index}",
        task_family="convfinqa_turn",
        capability_tags=("conversation",),
        question=f"q{index}",
        context=SampleContext(text=("filing",)),
        gold=GoldAnswer(answer=str(index), answer_type=AnswerType.NUMERIC, numeric_value=index),
        evaluation=EvaluationSpec(),
        source=SourceInfo(
            license="MIT", url="https://github.com/czyssrs/ConvFinQA", redistributable=True
        ),
        metadata={
            "conversation_id": conversation,
            "turn_index": str(index),
            "reuses_turns": ",".join(str(t) for t in reuses),
            "reuses_prior_answer": "true" if reuses else "false",
        },
    )


def _results(samples: list[CanonicalSample], passed: list[bool]) -> dict[str, MetricResult]:
    return {
        sample.sample_id: MetricResult(
            sample_id=sample.sample_id,
            metric_name="convfinqa_turn_accuracy",
            value=ok,
            passed=ok,
        )
        for sample, ok in zip(samples, passed, strict=True)
    }


# --------------------------------------------------------------------------- the honest headline


def test_full_conversation_accuracy_is_all_or_nothing() -> None:
    """The number a user actually experiences.

    Three turns at 67 % each is not "mostly right" — it is one conversation that went wrong
    somewhere, and the per-turn average hides that. Here two conversations of three turns score 4/6
    on turns (0.67) and 0/2 on conversations, because each had exactly one bad turn.
    """
    samples = [_turn("a", i) for i in range(3)] + [_turn("b", i) for i in range(3)]
    results = _results(samples, [True, False, True, True, True, False])

    analysis = analyze_conversations(samples, results, PROTOCOL)
    assert analysis is not None
    assert analysis.n_conversations == 2
    assert analysis.n_turns == 6
    assert analysis.turn_accuracy == pytest.approx(4 / 6)
    assert analysis.full_conversation_accuracy == 0.0


def test_a_conversation_that_is_right_all_the_way_through_counts() -> None:
    samples = [_turn("a", i) for i in range(3)] + [_turn("b", i) for i in range(3)]
    results = _results(samples, [True, True, True, True, False, True])

    analysis = analyze_conversations(samples, results, PROTOCOL)
    assert analysis is not None
    assert analysis.full_conversation_accuracy == pytest.approx(0.5)


def test_the_first_error_turn_says_which_problem_the_model_has() -> None:
    """A model whose first error is always turn 0 cannot read the table. One whose first error is
    turn 4 cannot hold a conversation. Same accuracy, opposite fixes."""
    shallow = [_turn("a", i) for i in range(5)]
    deep = [_turn("b", i) for i in range(5)]

    early = analyze_conversations(shallow, _results(shallow, [False] * 5), PROTOCOL)
    late = analyze_conversations(deep, _results(deep, [True, True, True, True, False]), PROTOCOL)

    assert early is not None and late is not None
    assert early.mean_first_error_turn == 0.0
    assert late.mean_first_error_turn == 4.0


def test_accuracy_by_turn_index_shows_where_it_slips() -> None:
    samples = [_turn(c, i) for c in ("a", "b") for i in range(3)]
    results = _results(samples, [True, True, False, True, False, False])

    analysis = analyze_conversations(samples, results, PROTOCOL)
    assert analysis is not None
    assert analysis.accuracy_by_turn_index == {0: 1.0, 1: 0.5, 2: 0.0}


# --------------------------------------------------------------------------- error propagation


def test_propagation_is_measured_against_the_turns_actually_consumed_not_mere_adjacency() -> None:
    """The distinction the whole module rests on.

    Turn 3 here is built on turn 0 (``reuses=(0,)``) and has nothing to do with turn 2. If the model
    got turn 0 right, turn 3's inputs are clean — *even if turn 2 was wrong*. Counting turn 2's
    failure against turn 3 would attribute an error to a value turn 3 never used, and the resulting
    "propagation rate" would be measuring the conversation's difficulty, not its contamination.

    So: turn 0 right, turn 2 wrong, turn 3 depends only on turn 0 → turn 3 is CLEAN.
    """
    samples = [
        _turn("a", 0),
        _turn("a", 1),
        _turn("a", 2),
        _turn("a", 3, reuses=(0,)),
    ]
    results = _results(samples, [True, True, False, True])

    analysis = analyze_conversations(samples, results, PROTOCOL)
    assert analysis is not None
    assert analysis.n_dependent_turns == 1
    assert analysis.n_poisoned_turns == 0
    assert analysis.accuracy_given_clean_inputs == 1.0
    assert analysis.accuracy_given_poisoned_inputs is None


def test_a_wrong_source_turn_poisons_the_turn_that_uses_it() -> None:
    """Turn 0 wrong → turn 2 (which consumes turn 0's answer) is running on a poisoned input. Under
    ``model_history`` that wrong number is literally in turn 2's prompt."""
    samples = [_turn("a", 0), _turn("a", 1), _turn("a", 2, reuses=(0, 1))]
    results = _results(samples, [False, True, False])

    analysis = analyze_conversations(samples, results, PROTOCOL)
    assert analysis is not None
    assert analysis.n_poisoned_turns == 1
    assert analysis.accuracy_given_poisoned_inputs == 0.0
    assert analysis.recovery_rate == 0.0


def test_recovery_is_recorded_when_the_model_gets_it_right_despite_a_bad_input() -> None:
    """The behaviour that stops one wrong answer becoming four: the model either noticed, or went
    back to the table instead of trusting the conversation."""
    samples = [_turn("a", 0), _turn("a", 1, reuses=(0,))]
    results = _results(samples, [False, True])

    analysis = analyze_conversations(samples, results, PROTOCOL)
    assert analysis is not None
    assert analysis.recovery_rate == 1.0
    assert analysis.accuracy_given_poisoned_inputs == 1.0


def test_the_propagation_effect_is_the_gap_between_clean_and_poisoned_inputs() -> None:
    """Conversation 'a': turn 0 right, so turn 1's input is clean, and it is right → clean = 1.0.
    Conversation 'b': turn 0 wrong, so turn 1's input is poisoned, and it is wrong → poisoned = 0.0.
    Effect = 1.0. Under model_history that is the cost of the model's own earlier mistake."""
    samples = [
        _turn("a", 0),
        _turn("a", 1, reuses=(0,)),
        _turn("b", 0),
        _turn("b", 1, reuses=(0,)),
    ]
    results = _results(samples, [True, True, False, False])

    analysis = analyze_conversations(samples, results, PROTOCOL)
    assert analysis is not None
    assert analysis.accuracy_given_clean_inputs == 1.0
    assert analysis.accuracy_given_poisoned_inputs == 0.0
    assert analysis.propagation_effect == 1.0


def test_context_loss_compares_dependent_turns_against_standalone_ones() -> None:
    """How much the model loses when a turn needs the conversation rather than just the document."""
    samples = [_turn("a", 0), _turn("a", 1), _turn("a", 2, reuses=(0,)), _turn("a", 3, reuses=(1,))]
    results = _results(samples, [True, True, False, False])

    analysis = analyze_conversations(samples, results, PROTOCOL)
    assert analysis is not None
    assert analysis.independent_turn_accuracy == 1.0
    assert analysis.dependent_turn_accuracy == 0.0
    assert analysis.context_loss == 1.0


# --------------------------------------------------------------------------- refusals to guess


def test_a_source_turn_missing_from_the_run_is_neither_clean_nor_poisoned() -> None:
    """A truncated conversation must not have its missing turns read as *correct*.

    Doing so would quietly move dependent turns into the clean bucket and make the model look like
    it handles good inputs worse than it does — flattering the propagation effect by diluting the
    thing it is measured against.
    """
    samples = [_turn("a", 3, reuses=(0,))]  # turn 0 is not in this run at all
    results = _results(samples, [False])

    analysis = analyze_conversations(samples, results, PROTOCOL)
    assert analysis is not None
    assert analysis.n_dependent_turns == 1
    assert analysis.n_poisoned_turns == 0
    assert analysis.accuracy_given_clean_inputs is None
    assert analysis.accuracy_given_poisoned_inputs is None
    assert analysis.propagation_effect is None


def test_a_benchmark_with_no_conversations_gets_no_conversation_report() -> None:
    """A FinQA run must not acquire an empty conversation analysis. Zeros in a report read as a
    measurement; ``None`` reads as "not measured", which is the truth."""
    from tests.factories import make_sample

    samples = [make_sample(sample_id=f"smoke:dev:{i}") for i in range(3)]
    results = _results(samples, [True, True, False])
    assert analyze_conversations(samples, results, PROTOCOL) is None


def test_an_ungraded_turn_is_not_counted_as_correct() -> None:
    """``passed=None`` means the metric did not apply. It is not a pass."""
    samples = [_turn("a", 0), _turn("a", 1)]
    results = {
        samples[0].sample_id: MetricResult(
            sample_id=samples[0].sample_id,
            metric_name="convfinqa_turn_accuracy",
            value=None,
            passed=None,
        )
    }
    analysis = analyze_conversations(samples, results, PROTOCOL)
    assert analysis is not None
    assert analysis.turn_accuracy == 0.0
    assert analysis.full_conversation_accuracy == 0.0
