"""Conversation-level analysis: what a per-turn score cannot see.

Turn accuracy tells you how often the model was right. It cannot tell you *why* it was wrong at turn
4, and in a conversation that is usually the only question worth asking — because the answer is
often "because it was wrong at turn 1, and turn 4 is built on turn 1".

Everything here is computed from two things the per-turn metrics already produce (which turns
passed) and one thing the adapter recovered from the gold programs (which turns each turn's answer
is actually *built on* — see ``datasets/convfinqa/adapter._reused_prior_turns``). Nothing here needs
a judge, and nothing here is a guess.

**The number this module exists to produce is the propagation effect**, and it only means anything
when the same model is run under both protocols:

- Under ``gold_history`` the model is handed the *gold* prior answers, so its own turn-1 mistake is
  nowhere in its turn-4 prompt. Whatever correlation remains between a wrong turn 1 and a wrong turn
  4 is just difficulty — hard conversations are hard throughout.
- Under ``model_history`` the model is handed its *own* prior answers, so a wrong turn 1 is sitting
  in turn 4's context, stated as fact.

The gap between the two is the part attributable to the conversation itself. That is why the
protocols are never mixed into one score: averaged together they would cancel exactly the effect
they exist to isolate.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from financebench.schemas.common import ConversationProtocol
from financebench.schemas.metric import MetricResult
from financebench.schemas.sample import CanonicalSample

__all__ = ["ConversationAnalysis", "analyze_conversations"]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class ConversationAnalysis:
    """What happened across the turns of a conversation, rather than within any one of them."""

    protocol: ConversationProtocol
    n_conversations: int
    n_turns: int

    turn_accuracy: float | None
    """Every turn, pooled. The number a single-turn benchmark would report."""

    full_conversation_accuracy: float | None
    """Fraction of conversations in which **every** turn was right.

    The honest headline for a multi-turn task, and always the bleakest: five turns at 70 % each is
    17 % if the errors are independent. A user does not experience five separate 70 %s — they
    experience one conversation that went wrong somewhere.
    """

    accuracy_by_turn_index: dict[int, float]
    """Accuracy at turn 0, turn 1, turn 2… — where in a conversation the model starts to slip."""

    mean_first_error_turn: float | None
    """Among conversations that went wrong at all, how deep they got first. A model whose first
    error is always at turn 0 has a *comprehension* problem; one whose first error is at turn 4 has
    a *memory* problem, and they need opposite fixes."""

    dependent_turn_accuracy: float | None
    """Turns whose answer is built on an earlier turn's answer."""

    independent_turn_accuracy: float | None
    """Turns that could in principle be answered from the table alone."""

    context_loss: float | None
    """``independent - dependent``. How much the model loses when a turn requires the conversation
    rather than just the document."""

    accuracy_given_clean_inputs: float | None
    """Dependent turns whose source turns the model got **right**."""

    accuracy_given_poisoned_inputs: float | None
    """Dependent turns where the model had already got at least one source turn **wrong**.

    Under ``model_history`` that wrong answer is literally in the prompt. Under ``gold_history`` it
    is not — the model is shown the gold — so this figure is the difficulty baseline that
    ``model_history`` must be measured against.
    """

    propagation_effect: float | None
    """``accuracy_given_clean_inputs - accuracy_given_poisoned_inputs``. Under ``model_history``,
    the cost of the model's own earlier mistake. Subtract the same figure computed under
    ``gold_history`` to remove the difficulty baseline and leave the propagation itself."""

    recovery_rate: float | None
    """Fraction of poisoned-input turns the model got right *anyway* — it noticed, or recomputed
    from the table instead of trusting the conversation. Under ``model_history`` this is the only
    behaviour that stops one bad answer becoming four."""

    n_dependent_turns: int
    n_poisoned_turns: int

    def to_json(self) -> dict[str, object]:
        return {
            "protocol": self.protocol.value,
            "n_conversations": self.n_conversations,
            "n_turns": self.n_turns,
            "turn_accuracy": self.turn_accuracy,
            "full_conversation_accuracy": self.full_conversation_accuracy,
            "accuracy_by_turn_index": {str(k): v for k, v in self.accuracy_by_turn_index.items()},
            "mean_first_error_turn": self.mean_first_error_turn,
            "dependent_turn_accuracy": self.dependent_turn_accuracy,
            "independent_turn_accuracy": self.independent_turn_accuracy,
            "context_loss": self.context_loss,
            "accuracy_given_clean_inputs": self.accuracy_given_clean_inputs,
            "accuracy_given_poisoned_inputs": self.accuracy_given_poisoned_inputs,
            "propagation_effect": self.propagation_effect,
            "recovery_rate": self.recovery_rate,
            "n_dependent_turns": self.n_dependent_turns,
            "n_poisoned_turns": self.n_poisoned_turns,
        }


def _turn_index(sample: CanonicalSample) -> int:
    try:
        return int(sample.metadata.get("turn_index", "0"))
    except ValueError:
        return 0


def _sources(sample: CanonicalSample) -> tuple[int, ...]:
    raw = sample.metadata.get("reuses_turns", "")
    return tuple(int(part) for part in raw.split(",") if part.strip().isdigit())


def analyze_conversations(
    samples: Sequence[CanonicalSample],
    results: Mapping[str, MetricResult],
    protocol: ConversationProtocol,
) -> ConversationAnalysis | None:
    """Roll per-turn outcomes up into a conversation-level picture.

    ``results`` maps sample_id → the preferred metric result for that turn. Returns ``None`` when no
    sample carries a conversation, so a FinQA run does not acquire an empty conversation report that
    reads as a measurement.
    """
    turns = [s for s in samples if s.metadata.get("conversation_id")]
    if not turns:
        return None

    by_conversation: dict[str, list[CanonicalSample]] = defaultdict(list)
    for sample in turns:
        by_conversation[sample.metadata["conversation_id"]].append(sample)
    for group in by_conversation.values():
        group.sort(key=_turn_index)

    def correct(sample: CanonicalSample) -> bool:
        result = results.get(sample.sample_id)
        # `passed is None` means the metric did not apply. It is not a pass, and it is not a
        # failure either — but a turn nobody graded cannot be counted as right.
        return bool(result is not None and result.passed)

    all_scores = [1.0 if correct(s) else 0.0 for s in turns]

    by_index: dict[int, list[float]] = defaultdict(list)
    for sample in turns:
        by_index[_turn_index(sample)].append(1.0 if correct(sample) else 0.0)

    complete: list[float] = []
    first_errors: list[float] = []
    for group in by_conversation.values():
        outcomes = [correct(sample) for sample in group]
        complete.append(1.0 if all(outcomes) else 0.0)
        for position, ok in enumerate(outcomes):
            if not ok:
                first_errors.append(float(_turn_index(group[position])))
                break

    # -- dependency: which turns are actually built on an earlier turn's answer -------------------
    dependent: list[float] = []
    independent: list[float] = []
    clean: list[float] = []
    poisoned: list[float] = []

    for group in by_conversation.values():
        outcome_by_turn = {_turn_index(sample): correct(sample) for sample in group}
        for sample in group:
            score = 1.0 if correct(sample) else 0.0
            sources = _sources(sample)
            if not sources:
                independent.append(score)
                continue
            dependent.append(score)
            # A source turn that isn't in this run (a truncated conversation) cannot be judged
            # clean or poisoned, so it is not treated as either — silently reading "absent" as
            # "correct" would move turns into the clean bucket and flatter the propagation figure.
            known = [outcome_by_turn[t] for t in sources if t in outcome_by_turn]
            if len(known) != len(sources):
                continue
            (clean if all(known) else poisoned).append(score)

    clean_accuracy = _mean(clean)
    poisoned_accuracy = _mean(poisoned)
    dependent_accuracy = _mean(dependent)
    independent_accuracy = _mean(independent)

    effect = (
        clean_accuracy - poisoned_accuracy
        if clean_accuracy is not None and poisoned_accuracy is not None
        else None
    )
    loss = (
        independent_accuracy - dependent_accuracy
        if independent_accuracy is not None and dependent_accuracy is not None
        else None
    )

    return ConversationAnalysis(
        protocol=protocol,
        n_conversations=len(by_conversation),
        n_turns=len(turns),
        turn_accuracy=_mean(all_scores),
        full_conversation_accuracy=_mean(complete),
        accuracy_by_turn_index={
            index: sum(scores) / len(scores) for index, scores in sorted(by_index.items())
        },
        mean_first_error_turn=_mean(first_errors),
        dependent_turn_accuracy=dependent_accuracy,
        independent_turn_accuracy=independent_accuracy,
        context_loss=loss,
        accuracy_given_clean_inputs=clean_accuracy,
        accuracy_given_poisoned_inputs=poisoned_accuracy,
        propagation_effect=effect,
        recovery_rate=poisoned_accuracy,
        n_dependent_turns=len(dependent),
        n_poisoned_turns=len(poisoned),
    )
