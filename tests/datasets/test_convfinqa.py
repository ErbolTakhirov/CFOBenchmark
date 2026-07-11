"""ConvFinQA adapter: real conversations, honestly labelled, with their dependencies recovered.

Run against six real dev conversations committed as a fixture (27 turns), so this suite says
something in CI and not only on the one machine with the 421-conversation download on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from financebench.datasets.base import create_dataset
from financebench.datasets.convfinqa.adapter import ConvFinQAAdapter, _reused_prior_turns
from financebench.evaluation.benchmark_metrics import metrics_for_run, preferred_metric_name
from financebench.schemas.manifest import AdapterStatus
from financebench.schemas.sample import CanonicalSample
from financebench.utils.errors import DatasetLoadError

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "convfinqa"


def _samples() -> list[CanonicalSample]:
    return list(ConvFinQAAdapter(data_dir=FIXTURE_DIR).load("dev"))


# --------------------------------------------------------------------------- registration


def test_the_adapter_is_registered() -> None:
    assert isinstance(create_dataset("convfinqa"), ConvFinQAAdapter)


def test_the_direct_answer_metric_is_ours_and_the_program_metrics_are_theirs() -> None:
    """ConvFinQA's official evaluator only ever grades programs. Reporting "program accuracy: 0.0"
    for a run that asked for a number would be a lie about the model, so a direct-answer run gets a
    metric that says in its name that it is not the official one."""
    assert (
        preferred_metric_name("convfinqa", "structured_financial_v1") == "convfinqa_turn_accuracy"
    )
    assert preferred_metric_name("convfinqa", "program_v1") == "convfinqa_execution_accuracy"

    names = {m.name for m in metrics_for_run("convfinqa", "program_v1")}
    assert "convfinqa_program_accuracy" in names


# --------------------------------------------------------------------------- conversations, intact


def test_turns_are_not_flattened_into_independent_questions() -> None:
    """The tempting shortcut, and it destroys the only thing this dataset measures.

    Turn 1 of a real conversation is *"and what was it in 2005?"* — a question with no meaning
    whatsoever on its own. Flattened into a standalone QA pair it is unanswerable by construction,
    and a model would be scored for failing to read the examiner's mind.
    """
    samples = _samples()
    assert len(samples) == 27

    conversations = {s.metadata["conversation_id"] for s in samples}
    assert len(conversations) == 6

    followups = [s for s in samples if s.metadata["turn_index"] != "0"]
    assert followups, "a conversation benchmark with no follow-up turns is not one"
    assert all(s.context.conversation_history for s in followups)


def test_every_turn_carries_the_table_and_the_text() -> None:
    """A turn's question is elliptical; its evidence is not. The model needs the filing at every
    turn, not just the first."""
    for sample in _samples():
        assert sample.context.tables, f"{sample.sample_id} lost its table"
        assert sample.context.text, f"{sample.sample_id} lost its surrounding text"


def test_the_gold_program_is_preserved_so_official_program_accuracy_is_computable() -> None:
    with_programs = [s for s in _samples() if s.gold.program]
    assert len(with_programs) == 27, "every turn has a gold program in ConvFinQA"
    assert any("(" in (s.gold.program or "") for s in with_programs), "some turn must compute"


def test_yes_no_turns_are_typed_as_text_not_forced_into_a_number() -> None:
    """ConvFinQA's ``greater`` op yields yes/no. Coercing that to a float would make the turn
    unscoreable, and a model that answered correctly would be marked wrong."""
    for sample in _samples():
        if sample.gold.numeric_value is None:
            assert sample.gold.answer.strip() != ""


# --------------------------------------------------------------------------- the dependency graph


def test_a_turns_program_operands_reveal_which_earlier_turns_it_is_built_on() -> None:
    """The signal that makes error propagation measurable.

    ConvFinQA's turn programs are self-contained — turn 4's is
    ``subtract(60.94, 25.14), divide(#0, 25.14)``, which recomputes rather than referring back — so
    there is no explicit cross-turn link anywhere in the data. But ``60.94`` and ``25.14`` *are* the
    answers to turns 0 and 1, and that is recoverable.
    """
    # dev record 0: answers 60.94, 25.14, 35.8, 25.14, 1.42403
    prior = [60.94, 25.14, 35.8, 25.14]
    assert _reused_prior_turns("subtract(60.94, 25.14), divide(#0, 25.14)", prior) == (0, 1, 3)


def test_a_reference_to_an_earlier_step_is_not_a_reference_to_an_earlier_turn() -> None:
    """``#0`` points at the first operation of the *same* program. Reading it as "turn 0" would
    invent a dependency on every multi-step turn in the dataset."""
    assert _reused_prior_turns("divide(#0, 100)", [0.0, 5.0]) == ()


def test_a_constant_is_not_mistaken_for_a_prior_answer() -> None:
    """``const_100`` is arithmetic, not memory. If a prior turn's answer happened to be 100, a
    turn that merely divides by a hundred would otherwise be recorded as depending on it."""
    assert _reused_prior_turns("divide(#0, const_100)", [100.0]) == ()


def test_the_first_turn_can_depend_on_nothing() -> None:
    assert _reused_prior_turns("60.94", []) == ()


def test_real_conversations_contain_both_dependent_and_independent_turns() -> None:
    """If every turn were dependent — or none were — the context-loss and propagation figures would
    have nothing to compare against and would silently be ``None``."""
    samples = _samples()
    dependent = [s for s in samples if s.metadata["reuses_prior_answer"] == "true"]
    independent = [s for s in samples if s.metadata["reuses_prior_answer"] == "false"]

    assert dependent, "no turn was found to build on an earlier one — the analysis has no subject"
    assert independent, "no standalone turn — context loss would have no baseline"


def test_a_turn_never_depends_on_itself_or_on_a_later_turn() -> None:
    """Causality. A dependency on a *later* turn would mean the analysis was reading the future,
    and every propagation number would be nonsense."""
    for sample in _samples():
        index = int(sample.metadata["turn_index"])
        sources = [int(t) for t in sample.metadata["reuses_turns"].split(",") if t]
        assert all(source < index for source in sources), f"{sample.sample_id} depends on {sources}"


# --------------------------------------------------------------------------- honest labelling


def test_the_test_split_is_refused_rather_than_faked() -> None:
    """ConvFinQA's test gold was never released — it lives behind a CodaLab submission. A locally
    re-derived "test" split would produce numbers that look comparable with published ones and are
    not."""
    with pytest.raises(DatasetLoadError, match="never publicly released"):
        ConvFinQAAdapter(data_dir=FIXTURE_DIR).load("test")


def test_the_manifest_says_partial_and_says_why() -> None:
    manifest = ConvFinQAAdapter().manifest()
    assert manifest.status is AdapterStatus.PARTIAL
    assert manifest.license == "MIT"

    limitations = " ".join(manifest.known_limitations)
    assert "CodaLab" in limitations
    assert "gold_history" in limitations and "model_history" in limitations
    assert "never mixed" in limitations
