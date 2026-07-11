"""Gold leakage through the conversation history — a channel the main leakage suite cannot see.

``tests/security/test_gold_answer_leakage.py`` proves that scrubbing ``sample.gold`` leaves the
rendered request byte-identical. For ConvFinQA that proof is **true and insufficient**, and the gap
is worth being precise about:

A turn's prior conversation lives in ``sample.context.conversation_history``, not in ``sample.gold``.
Scrubbing the gold does not touch it, so scrub-equivalence would hold just as happily if the current
turn's answer had been accidentally appended to its own history. The strongest test in the suite
would report no leak while the model was being handed the answer.

That is not hypothetical bookkeeping — it is the *one* benchmark where gold is meant to be in the
prompt. Under ``gold_history`` the prior turns' gold answers legitimately appear: that IS the
protocol, and refusing them would make the benchmark measure something else. So the boundary is not
"no gold in the prompt" but "**gold up to turn N-1, and never turn N**", and only a positional check
can state it.

Why positional and not by value: turn 0's gold answer is often a number lifted straight out of the
table (*"what was the price in 2007?"* → ``60.94``, which is a cell), and two turns of one
conversation frequently share an answer (in dev record 0, turn 1 and turn 3 are both ``25.14``).
Grepping the prompt for the gold string would therefore fire constantly on correct behaviour. The
property that actually holds is structural: the history is exactly the prior turns, in order, and
nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from financebench.datasets.convfinqa.adapter import ConvFinQAAdapter
from financebench.execution.engine import with_model_history
from financebench.prompts.renderer import render_messages
from financebench.schemas.sample import CanonicalSample

#: Six real dev conversations (27 turns), committed so this suite runs everywhere — including CI,
#: where the full download does not exist. A leakage guarantee that only holds on the one machine
#: with the dataset on disk is not a guarantee. ConvFinQA is MIT, so redistributing a sample is fine.
FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "convfinqa"
DATA_DIR = Path("data/downloads/convfinqa")


def _conversations(limit: int = 40) -> list[CanonicalSample]:
    return list(ConvFinQAAdapter(data_dir=FIXTURE_DIR).load("dev"))[:limit]


# ------------------------------------------------------------------ gold_history: N-1, never N


def test_the_history_is_exactly_the_prior_turns_and_nothing_else() -> None:
    """The boundary, stated positionally because it cannot be stated by value.

    If the current turn's answer were ever appended to its own history, this is the test that
    catches it — and it is the *only* one that can, because scrub-equivalence cannot see a channel
    that does not live on ``sample.gold``.
    """
    by_conversation: dict[str, dict[int, CanonicalSample]] = {}
    for sample in _conversations(limit=120):
        conversation = sample.metadata["conversation_id"]
        by_conversation.setdefault(conversation, {})[int(sample.metadata["turn_index"])] = sample

    checked = 0
    for turns in by_conversation.values():
        for index, sample in sorted(turns.items()):
            history = sample.context.conversation_history

            # One user + one assistant per PRIOR turn. Not one more.
            assert len(history) == 2 * index, (
                f"turn {index} carries {len(history)} history entries; "
                f"{2 * index} prior turns means exactly {2 * index} entries"
            )

            for prior in range(index):
                question, answer = history[2 * prior], history[2 * prior + 1]
                assert question.role == "user"
                assert answer.role == "assistant"
                assert question.content == turns[prior].question
                assert answer.content == turns[prior].gold.answer
                checked += 1

    assert checked > 0, "no multi-turn conversation was examined — this test proved nothing"


def test_the_current_turns_gold_derivation_is_never_in_its_own_prompt() -> None:
    """The answer may be a table cell, but the *derivation* never is.

    ``subtract(60.94, 25.14), divide(#0, 25.14)`` does not occur naturally in a 10-K. If the current
    turn's gold program reached the prompt, the question would stop being a question.

    Only turns with a real operation are checked, and that exclusion is the data talking rather than
    a convenience: a *lookup* turn's "program" is the bare literal ``60.94``, which is not a
    derivation at all — it is a cell of the table the model is supposed to be reading. Asserting its
    absence would demand that the evidence be withheld from an evidence question. (This test failed
    on exactly that, which is how the distinction got noticed.)
    """
    checked = 0
    for sample in _conversations(limit=120):
        program = sample.gold.program or ""
        if "(" not in program:  # a bare literal — a lookup, not a derivation
            continue
        prompt = "\n".join(m.content for m in render_messages(sample))
        assert program not in prompt
        checked += 1
    assert checked > 0, "no sample carried a computed program — this test proved nothing"


def test_a_history_turns_gold_program_is_carried_but_never_rendered() -> None:
    """``ConversationTurn`` has ``turn_program`` and ``turn_answer`` fields, and they hold gold.

    They exist for the evaluator. Nothing stops a future renderer from formatting the whole turn
    object into the prompt, at which point every conversation would ship its own worked solution —
    so this pins the fact that only ``role`` and ``content`` are ever rendered.
    """
    sample = next(s for s in _conversations(limit=120) if s.context.conversation_history)
    poisoned = sample.context.model_copy(
        update={
            "conversation_history": tuple(
                turn.model_copy(
                    update={
                        "turn_program": "LEAKED_PROGRAM_SENTINEL",
                        "turn_answer": "LEAKED_ANSWER_SENTINEL",
                    }
                )
                for turn in sample.context.conversation_history
            )
        }
    )
    prompt = "\n".join(
        m.content for m in render_messages(sample.model_copy(update={"context": poisoned}))
    )
    assert "LEAKED_PROGRAM_SENTINEL" not in prompt
    assert "LEAKED_ANSWER_SENTINEL" not in prompt


# ------------------------------------------------------------------ model_history: no gold at all


def test_under_model_history_no_gold_answer_reaches_the_prompt_at_all() -> None:
    """The whole point of the protocol.

    Under ``gold_history`` the prior gold answers are in the prompt by design. Under
    ``model_history`` they must be *gone* — replaced, every one of them, by what the model itself
    said. A single surviving gold answer would repair the chain at exactly the point the protocol
    exists to watch it break.
    """
    sample = next(s for s in _conversations(limit=120) if len(s.context.conversation_history) >= 4)
    slots = sum(1 for t in sample.context.conversation_history if t.role == "assistant")
    spoken = [f"MODEL_SAID_{i}" for i in range(slots)]

    rewritten = with_model_history(sample, spoken)
    prompt = "\n".join(m.content for m in render_messages(rewritten))

    gold_in_history = [
        turn.content for turn in sample.context.conversation_history if turn.role == "assistant"
    ]
    history_block = prompt.split("Context:")[0]
    for gold in gold_in_history:
        assert gold not in history_block, (
            f"the gold answer {gold!r} survived into a model_history prompt — the model is being "
            "handed the very answer this protocol withholds"
        )
    for said in spoken:
        assert said in prompt

    # And the gold that the evaluator needs is still on the sample, untouched.
    assert rewritten.gold == sample.gold


def test_model_history_turns_carry_no_gold_fields() -> None:
    """The rewritten assistant turns must be *bare*: content only. Copying the gold ``turn_program``
    across while replacing the content would leave the derivation in place — the answer withheld and
    the working shown."""
    sample = next(s for s in _conversations(limit=120) if s.context.conversation_history)
    slots = sum(1 for t in sample.context.conversation_history if t.role == "assistant")
    rewritten = with_model_history(sample, ["1.0"] * slots)

    for turn in rewritten.context.conversation_history:
        if turn.role == "assistant":
            assert turn.turn_program is None
            assert turn.turn_answer is None


def test_the_questions_are_not_rewritten_only_the_answers() -> None:
    """The questions are the dataset, not a prediction. A protocol that replaced them would be
    evaluating a conversation the model invented."""
    sample = next(s for s in _conversations(limit=120) if len(s.context.conversation_history) >= 4)
    slots = sum(1 for t in sample.context.conversation_history if t.role == "assistant")
    rewritten = with_model_history(sample, [f"MODEL_SAID_{i}" for i in range(slots)])

    original_questions = [
        t.content for t in sample.context.conversation_history if t.role == "user"
    ]
    rewritten_questions = [
        t.content for t in rewritten.context.conversation_history if t.role == "user"
    ]
    assert rewritten_questions == original_questions


def test_a_turn_with_no_history_is_untouched_so_both_protocols_share_its_cache_entry() -> None:
    """Turn 0 has no history, so the two protocols ask it *the same question* — and should not pay
    for the same inference twice. This is a real saving on a 1,490-turn benchmark, and it is only
    safe because the prompt is genuinely identical."""
    turn_zero = next(s for s in _conversations() if s.metadata["turn_index"] == "0")
    assert with_model_history(turn_zero, []) is turn_zero


def test_the_fixture_really_contains_multi_turn_conversations() -> None:
    """Guards the guard: if the adapter silently produced no multi-turn samples, every test above
    would pass vacuously — and a vacuous leakage suite is worse than none, because it reads as
    proof."""
    samples = _conversations(limit=200)
    assert len(samples) == 27
    assert len({s.metadata["conversation_id"] for s in samples}) == 6
    assert any(len(s.context.conversation_history) >= 6 for s in samples)


@pytest.mark.skipif(
    not (DATA_DIR / "dev.json").is_file(),
    reason="the full convfinqa download is not present; the fixture suite above still ran",
)
def test_the_full_dataset_matches_what_the_manifest_claims() -> None:
    """And when the real download IS present, the numbers in the manifest must be the numbers on
    disk. 421 conversations, 1,490 turns — claimed in docs, asserted here."""
    raw = json.loads((DATA_DIR / "dev.json").read_text(encoding="utf-8"))
    assert len(raw) == 421
    turns = sum(len(r["annotation"]["dialogue_break"]) for r in raw)
    assert turns == 1490
    assert len(list(ConvFinQAAdapter(data_dir=DATA_DIR).load("dev"))) == turns
