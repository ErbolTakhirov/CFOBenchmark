"""The two conversation protocols, and the engine that has to keep them apart.

A conversation is the one place in this platform where a sample is **not** independent of its
neighbours, and that breaks two assumptions the engine was built on: that every prompt can be
constructed before any inference happens, and that samples can be fired off in any order. Under
``model_history`` turn 3's prompt literally contains the model's answer to turn 2, which does not
exist yet.

These tests pin the behaviour that falls out of that:

- turns of one conversation run **in order**, and conversations run **concurrently** with each other;
- ``gold_history`` shows the model the gold prior answers, ``model_history`` shows it its own;
- a run that cannot honour ``model_history`` **fails loudly** instead of quietly substituting gold.

The last one is the one that matters. Substituting gold would be invisible, plausible, and would
inflate exactly the score whose entire purpose is to expose what happens when the model has to live
with its own mistakes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from financebench.execution.cache import ResponseCache
from financebench.execution.engine import (
    RunEngine,
    build_request,
    conversation_groups,
    model_answer_text,
    with_model_history,
)
from financebench.models.base import ModelProvider
from financebench.schemas.common import AnswerType, ConversationProtocol, SplitOrigin
from financebench.schemas.model_io import (
    FinancialAnswer,
    ModelRequest,
    ModelResponse,
    ModelSpec,
)
from financebench.schemas.prediction import Prediction
from financebench.schemas.run import RunConfig
from financebench.schemas.sample import (
    CanonicalSample,
    ConversationTurn,
    EvaluationSpec,
    GoldAnswer,
    SampleContext,
    SourceInfo,
)

MODEL = ModelSpec.parse("mock/scripted")


def _sid(conversation: str, index: int) -> str:
    """The id convention the schema enforces: ``{benchmark}:{split}:...``."""
    return f"convfinqa:dev:{conversation}#t{index}"


def _turn(conversation: str, index: int, n_turns: int = 3) -> CanonicalSample:
    """One turn of a synthetic conversation, shaped exactly as the ConvFinQA adapter shapes one."""
    history = tuple(
        entry
        for prior in range(index)
        for entry in (
            ConversationTurn(role="user", content=f"question {prior}"),
            ConversationTurn(
                role="assistant", content=f"GOLD_{prior}", turn_answer=f"GOLD_{prior}"
            ),
        )
    )
    return CanonicalSample(
        benchmark="convfinqa",
        benchmark_version="test",
        split="dev",
        split_origin=SplitOrigin.OFFICIAL,
        sample_id=_sid(conversation, index),
        task_family="convfinqa_turn",
        capability_tags=("conversation",),
        question=f"question {index}",
        context=SampleContext(text=("some filing text",), conversation_history=history),
        gold=GoldAnswer(
            answer=f"GOLD_{index}", answer_type=AnswerType.NUMERIC, numeric_value=float(index)
        ),
        evaluation=EvaluationSpec(),
        source=SourceInfo(
            license="MIT", url="https://github.com/czyssrs/ConvFinQA", redistributable=True
        ),
        metadata={
            "conversation_id": conversation,
            "turn_index": str(index),
            "n_turns": str(n_turns),
        },
    )


class _Recorder(ModelProvider):
    """Answers ``TURN_<n>`` and records the order calls arrived in, plus every prompt it saw."""

    name = "mock"

    def __init__(self) -> None:
        self.order: list[str] = []
        self.prompts: dict[str, str] = {}

    async def generate(self, request: ModelRequest) -> ModelResponse:
        sample_id = request.sample_id or ""
        self.order.append(sample_id)
        self.prompts[sample_id] = "\n".join(m.content for m in request.messages)
        await asyncio.sleep(0)  # a real await, so out-of-order completion is possible
        index = sample_id.rsplit("#t", 1)[-1]
        return ModelResponse(
            provider="mock",
            model="scripted",
            content="{}",
            financial_answer=FinancialAnswer(answer=f"TURN_{index}"),
            parsed=True,
        )

    async def aclose(self) -> None:
        return None


# --------------------------------------------------------------------------- grouping


def test_a_single_turn_benchmark_is_unaffected() -> None:
    """Every sample its own group: exactly the fully-parallel behaviour that existed before
    conversations did. Adding ConvFinQA must not slow FinQA down."""
    from tests.factories import make_sample

    samples = [make_sample(sample_id=f"smoke:dev:{i}") for i in range(5)]
    groups = conversation_groups(samples)
    assert [len(group) for group in groups] == [1, 1, 1, 1, 1]


def test_turns_are_grouped_by_conversation_and_ordered_by_turn_index() -> None:
    """Order comes from ``turn_index``, never from the order the samples arrived in.

    Trusting arrival order works until a stratified manifest or a shuffled split hands the turns over
    backwards — at which point the model is shown a conversation that runs in reverse, and nothing
    errors. It just scores badly, for a reason nobody would ever find.
    """
    shuffled = [_turn("a", 2), _turn("b", 0), _turn("a", 0), _turn("b", 1), _turn("a", 1)]
    groups = conversation_groups(shuffled)

    assert len(groups) == 2
    assert [s.sample_id for _, s in groups[0]] == [_sid("a", i) for i in range(3)]
    assert [s.sample_id for _, s in groups[1]] == [_sid("b", i) for i in range(2)]

    # Input positions are preserved, which is how predictions get put back in the run's order.
    assert sorted(index for group in groups for index, _ in group) == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- protocol behaviour


@pytest.mark.asyncio
async def test_gold_history_shows_the_model_the_gold_prior_answers(tmp_path: Path) -> None:
    provider = _Recorder()
    samples = [_turn("c", i) for i in range(3)]

    await RunEngine().run(
        samples=samples,
        model=MODEL,
        config=RunConfig(conversation_protocol=ConversationProtocol.GOLD_HISTORY),
        cache=ResponseCache(tmp_path),
        provider=provider,
    )
    prompt = provider.prompts[_sid("c", 2)]
    assert "GOLD_0" in prompt and "GOLD_1" in prompt
    assert "TURN_0" not in prompt, "the model's own answer must not appear under gold_history"


@pytest.mark.asyncio
async def test_model_history_shows_the_model_its_own_prior_answers(tmp_path: Path) -> None:
    """The measurement. Turn 2's prompt contains what the model actually said at turns 0 and 1 —
    right or wrong — and none of the gold."""
    provider = _Recorder()
    samples = [_turn("c", i) for i in range(3)]

    await RunEngine().run(
        samples=samples,
        model=MODEL,
        config=RunConfig(conversation_protocol=ConversationProtocol.MODEL_HISTORY),
        cache=ResponseCache(tmp_path),
        provider=provider,
    )
    prompt = provider.prompts[_sid("c", 2)]
    assert "TURN_0" in prompt and "TURN_1" in prompt
    assert "GOLD_0" not in prompt and "GOLD_1" not in prompt


@pytest.mark.asyncio
async def test_turns_run_in_order_within_a_conversation(tmp_path: Path) -> None:
    """Not an aesthetic preference. Under model_history, turn 2's prompt cannot be *built* until
    turn 1 has answered — ``asyncio.gather`` over all turns would construct a prompt containing an
    answer the model has not given yet."""
    provider = _Recorder()
    samples = [_turn("c", i) for i in range(4)]

    await RunEngine().run(
        samples=samples,
        model=MODEL,
        config=RunConfig(conversation_protocol=ConversationProtocol.MODEL_HISTORY, concurrency=8),
        cache=ResponseCache(tmp_path),
        provider=provider,
    )
    assert provider.order == [_sid("c", i) for i in range(4)]


@pytest.mark.asyncio
async def test_different_conversations_still_run_concurrently(tmp_path: Path) -> None:
    """Sequential *within* a conversation, parallel *across* — otherwise a 1,490-turn benchmark
    becomes a 1,490-request queue, and the ordering constraint costs far more than it buys."""
    provider = _Recorder()
    samples = [_turn(name, i) for name in ("a", "b", "c") for i in range(3)]

    await RunEngine().run(
        samples=samples,
        model=MODEL,
        config=RunConfig(conversation_protocol=ConversationProtocol.MODEL_HISTORY, concurrency=3),
        cache=ResponseCache(tmp_path),
        provider=provider,
    )
    # If conversations ran one after another, the order would be aaa bbb ccc. Interleaving proves
    # they advanced together.
    assert provider.order != [_sid(name, i) for name in ("a", "b", "c") for i in range(3)]
    for name in ("a", "b", "c"):
        turns = [s for s in provider.order if s.endswith(f":{name}#t0") or f":{name}#t" in s]
        assert turns == [_sid(name, i) for i in range(3)]


@pytest.mark.asyncio
async def test_predictions_come_back_in_input_order_however_the_conversations_finished(
    tmp_path: Path,
) -> None:
    """Conversations complete in whatever order they complete. The artifacts must not."""
    samples = [_turn(name, i) for name in ("a", "b") for i in range(3)]

    result = await RunEngine().run(
        samples=samples,
        model=MODEL,
        config=RunConfig(conversation_protocol=ConversationProtocol.MODEL_HISTORY),
        cache=ResponseCache(tmp_path),
        provider=_Recorder(),
    )
    assert [p.sample_id for p in result.predictions] == [s.sample_id for s in samples]


# --------------------------------------------------------------------------- the refusal to guess


@pytest.mark.asyncio
async def test_model_history_refuses_to_run_a_conversation_missing_its_opening_turns(
    tmp_path: Path,
) -> None:
    """The most important test here.

    Turn 2 needs the model's answers to turns 0 and 1. If those turns are not in the run, there are
    two options: quietly fall back to the *gold* prior answers, or refuse. Falling back is the
    tempting one — it always "works" — and it is the same bug as substituting the gold evidence text
    when retrieval finds nothing: it would repair the prompt precisely where the chain is broken, and
    the model_history score, whose whole purpose is to show what a wrong answer costs later, would be
    quietly propped up by the right ones.
    """
    from financebench.utils.errors import ConfigError

    orphan = [_turn("c", 2)]  # turn 2, with turns 0 and 1 nowhere in the run

    with pytest.raises(ConfigError, match="whole conversations"):
        await RunEngine().run(
            samples=orphan,
            model=MODEL,
            config=RunConfig(conversation_protocol=ConversationProtocol.MODEL_HISTORY),
            cache=ResponseCache(tmp_path),
            provider=_Recorder(),
        )


@pytest.mark.asyncio
async def test_gold_history_happily_runs_an_isolated_turn(tmp_path: Path) -> None:
    """Because under gold_history there is no chain to break — every turn stands alone against the
    gold conversation, which is exactly why it is the protocol that can be sampled."""
    result = await RunEngine().run(
        samples=[_turn("c", 2)],
        model=MODEL,
        config=RunConfig(conversation_protocol=ConversationProtocol.GOLD_HISTORY),
        cache=ResponseCache(tmp_path),
        provider=_Recorder(),
    )
    assert result.n_errors == 0


# --------------------------------------------------------------------------- what the model "said"


def test_a_failed_turn_is_reported_to_the_next_turn_as_a_failure_not_as_silence() -> None:
    """If turn 1 produced nothing, turn 2 must see that it produced nothing. Blanking it would hide
    the failure the protocol exists to propagate."""
    sample = _turn("c", 1)
    dead = Prediction(
        sample_id=sample.sample_id,
        benchmark="convfinqa",
        split="dev",
        request=build_request(sample, MODEL, RunConfig()),
        created_at="t",
        response=None,
        error="connection reset",
    )
    assert model_answer_text(dead) == "(no answer)"


def test_a_numeric_answer_is_passed_on_in_the_shape_the_gold_history_uses() -> None:
    """The gold history says ``25.14``, not ``{"answer": "25.14", "numeric_value": 25.14, ...}``.
    Feeding the raw JSON envelope forward would change the *format* of the conversation between the
    two protocols, and then their difference would no longer be attributable to the answers."""
    sample = _turn("c", 1)
    response = ModelResponse(
        provider="mock",
        model="x",
        content="{}",
        financial_answer=FinancialAnswer(answer="about 25.14 dollars", numeric_value=25.14),
        parsed=True,
    )
    prediction = Prediction(
        sample_id=sample.sample_id,
        benchmark="convfinqa",
        split="dev",
        request=build_request(sample, MODEL, RunConfig()),
        created_at="t",
        response=response,
    )
    assert model_answer_text(prediction) == "25.14"


def test_a_refusal_is_passed_on_as_a_refusal() -> None:
    sample = _turn("c", 1)
    prediction = Prediction(
        sample_id=sample.sample_id,
        benchmark="convfinqa",
        split="dev",
        request=build_request(sample, MODEL, RunConfig()),
        created_at="t",
        response=ModelResponse(
            provider="mock",
            model="x",
            content="{}",
            financial_answer=FinancialAnswer(answer="", insufficient_information=True),
            parsed=True,
        ),
    )
    assert model_answer_text(prediction) == "(insufficient information)"


def test_rewriting_history_leaves_the_gold_untouched() -> None:
    """The evaluator still needs it. Only the *prompt* is rewritten."""
    sample = _turn("c", 2)
    rewritten = with_model_history(sample, ["7", "8"])
    assert rewritten.gold == sample.gold
    assert [t.content for t in rewritten.context.conversation_history if t.role == "assistant"] == [
        "7",
        "8",
    ]
