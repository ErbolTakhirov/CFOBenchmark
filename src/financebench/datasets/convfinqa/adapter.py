"""ConvFinQA — real conversations, not independent questions dressed up as one.

The whole point of ConvFinQA is that a turn **does not stand alone**. Turn 1 of a real conversation
is *"and what was it in 2005?"* — a question with no meaning whatsoever without turn 0. Flattening
these into independent QA pairs, which is the tempting shortcut, destroys the only thing the dataset
measures.

So conversations are preserved: `conversation_id`, turn order, the table and text context, and every
prior question and answer.

**Two protocols, never mixed into one score:**

- ``gold_history`` — each turn is given the *gold* prior conversation. This isolates per-turn
  reasoning: the model cannot be wrong at turn 3 because it was wrong at turn 1. It is also the
  official setting, and it is the one comparable with published numbers.
- ``model_history`` — each turn is given the model's **own** prior answers. This is what a real
  conversation actually is, and it is the only way to measure **error propagation**: one wrong
  answer at turn 1 poisons every turn that refers back to it.

The difference between the two is the interesting number. A model that scores well under
`gold_history` and collapses under `model_history` is a model that cannot hold a conversation,
however well it can answer a question.

Under ``gold_history`` the prior turns' *gold* answers legitimately enter the prompt — that is the
protocol. The **current** turn's gold never does, and the leakage suite has an explicit, narrow
exemption saying exactly that.

Data: `czyssrs/ConvFinQA` @ `cf3eed2`, MIT. dev = 421 conversations / 1,490 turns, gold public.
The test split's gold was never released (CodaLab submission only), so it is not supported here.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from financebench.datasets.base import DatasetAdapter, register_dataset
from financebench.datasets.downloader import download_file
from financebench.schemas.common import AnswerType, SplitOrigin
from financebench.schemas.manifest import AdapterStatus, DatasetManifest
from financebench.schemas.sample import (
    CanonicalSample,
    ConversationTurn,
    EvaluationSpec,
    GoldAnswer,
    SampleContext,
    SourceInfo,
    Table,
)
from financebench.utils.errors import DatasetLoadError

__all__ = ["ConvFinQAAdapter"]

_PINNED_COMMIT = "cf3eed2d5984960bf06bb8145bcea5e80b0222a6"
_DATA_ZIP_URL = f"https://github.com/czyssrs/ConvFinQA/raw/{_PINNED_COMMIT}/data.zip"
_DEFAULT_DATA_DIR = Path("data/downloads/convfinqa")

#: Only splits whose gold is public. The test split's answers were never released.
_SPLITS = {"dev": "dev.json", "train": "train.json"}

#: A bare number in a turn's program. Deliberately excludes ``#0``-style references (which point at
#: an earlier step of the *same* program, not an earlier turn) and ``const_100``-style constants,
#: leaving only the values the program actually pulled out of the world.
_OPERAND = re.compile(r"(?<![#\w.])(-?\d+(?:\.\d+)?)")


def _operands(program: str) -> list[float]:
    stripped = re.sub(r"#\d+", "", re.sub(r"const_[\d_]+", "", program))
    values: list[float] = []
    for token in _OPERAND.findall(stripped):
        try:
            values.append(float(token))
        except ValueError:  # pragma: no cover — the regex only matches parseable numbers
            continue
    return values


def _reused_prior_turns(program: str, prior_answers: Sequence[Any]) -> tuple[int, ...]:
    """Which earlier turns' answers does this turn's program actually consume?

    This is the objective backbone of every conversation metric downstream, and it is worth being
    precise about what it is and is not.

    ConvFinQA's turn programs are *self-contained*: turn 4's program is
    ``subtract(60.94, 25.14), divide(#0, 25.14)`` — it recomputes the subtraction rather than
    referencing turn 2's result, so ``#0`` is an intra-program step reference and there is no
    explicit cross-turn link anywhere in the data. But ``60.94`` and ``25.14`` *are* the answers to
    turns 0 and 1, and turn 4's question — *"and how much does that change represent...?"* — cannot
    even be understood without them.

    So dependency is recovered from the numbers: a turn depends on an earlier turn when its program
    consumes a value that earlier turn produced. That is what makes error propagation measurable
    rather than assumed — under ``model_history``, if the model said 55.0 at turn 0, then the value
    turn 4 needs is *wrong in its own prompt*, and we can say exactly which earlier mistake it
    inherited instead of guessing from adjacency.

    It is a heuristic in one direction only: a value a prior turn produced is usually also a cell in
    the table, so an unrelated turn could in principle reuse the same number by coincidence. It
    cannot miss a real dependency, and that asymmetry is the right one — an over-count dilutes the
    propagation rate toward zero, so it errs against the interesting finding rather than toward it.
    """
    if not program:
        return ()
    values = _operands(program)
    if not values:
        return ()

    reused: list[int] = []
    for index, answer in enumerate(prior_answers):
        try:
            gold = float(answer)
        except (TypeError, ValueError):
            continue
        scale = max(abs(gold), 1.0)
        if any(abs(value - gold) <= 1e-4 * scale for value in values):
            reused.append(index)
    return tuple(reused)


@register_dataset("convfinqa")
class ConvFinQAAdapter(DatasetAdapter):
    name = "convfinqa"

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR

    def prepare(self) -> None:
        archive = self._data_dir / "data.zip"
        if not (self._data_dir / "dev.json").is_file():
            if not archive.is_file():
                download_file(_DATA_ZIP_URL, archive, max_bytes=100 * 1024 * 1024)
            with zipfile.ZipFile(archive) as zf:
                for member in zf.namelist():
                    name = Path(member).name
                    if name in _SPLITS.values():
                        (self._data_dir / name).write_bytes(zf.read(member))

    def load(self, split: str) -> Sequence[CanonicalSample]:
        if split not in _SPLITS:
            raise DatasetLoadError(
                f"convfinqa has no split {split!r}; available: {sorted(_SPLITS)}. "
                "(The test split's gold answers were never publicly released — they exist only "
                "behind a CodaLab submission — so it is not supported here rather than faked.)"
            )
        path = self._data_dir / _SPLITS[split]
        if not path.is_file():
            raise DatasetLoadError(
                f"convfinqa {split} not found at {path}. Run `financebench prepare convfinqa`."
            )
        records = json.loads(path.read_text(encoding="utf-8"))
        return list(self._to_samples(records, split))

    def _to_samples(
        self, records: Sequence[dict[str, Any]], split: str
    ) -> Iterator[CanonicalSample]:
        for record in records:
            yield from self._conversation(record, split)

    def _conversation(self, record: dict[str, Any], split: str) -> Iterator[CanonicalSample]:
        annotation = record.get("annotation") or {}
        questions: list[str] = annotation.get("dialogue_break") or []
        programs: list[str] = annotation.get("turn_program") or []
        answers: list[Any] = annotation.get("exe_ans_list") or []
        if not questions or len(questions) != len(answers):
            return

        conversation_id = str(record.get("id", "")).strip()
        if not conversation_id:
            return

        rows = tuple(tuple(str(cell) for cell in row) for row in (record.get("table") or []))
        tables = (Table(table_id="table", rows=rows, header_rows=1),) if rows else ()
        text = tuple(
            line
            for line in ((record.get("pre_text") or []) + (record.get("post_text") or []))
            if str(line).strip()
        )

        for turn_index, question in enumerate(questions):
            program = programs[turn_index] if turn_index < len(programs) else ""
            gold = answers[turn_index]
            reused = _reused_prior_turns(program, answers[:turn_index])

            numeric: float | None
            try:
                numeric = float(gold)
            except (TypeError, ValueError):
                numeric = None  # ConvFinQA's `greater` op yields yes/no

            # The GOLD prior conversation. Under model_history the engine replaces the assistant
            # turns with the model's own answers; under gold_history it uses these. Either way the
            # CURRENT turn's answer is never in here.
            history = tuple(
                turn
                for prior in range(turn_index)
                for turn in (
                    ConversationTurn(role="user", content=str(questions[prior])),
                    ConversationTurn(
                        role="assistant",
                        content=str(answers[prior]),
                        turn_program=(programs[prior] if prior < len(programs) else None),
                        turn_answer=str(answers[prior]),
                    ),
                )
            )

            yield CanonicalSample(
                benchmark="convfinqa",
                benchmark_version=f"official@{_PINNED_COMMIT[:8]}",
                split=split,
                split_origin=SplitOrigin.OFFICIAL,
                sample_id=f"convfinqa:{split}:{conversation_id}#t{turn_index}",
                task_family="convfinqa_turn",
                capability_tags=("conversation", "table_text", "calculation"),
                question=str(question),
                context=SampleContext(text=text, tables=tables, conversation_history=history),
                gold=GoldAnswer(
                    answer=str(gold),
                    answer_type=AnswerType.NUMERIC if numeric is not None else AnswerType.TEXT,
                    numeric_value=numeric,
                    program=program or None,
                ),
                evaluation=EvaluationSpec(absolute_tolerance=1e-3),
                source=SourceInfo(
                    license="MIT",
                    url=f"https://github.com/czyssrs/ConvFinQA/tree/{_PINNED_COMMIT}",
                    redistributable=True,
                ),
                metadata={
                    "conversation_id": conversation_id,
                    "turn_index": str(turn_index),
                    "n_turns": str(len(questions)),
                    "is_first_turn": "true" if turn_index == 0 else "false",
                    "is_last_turn": "true" if turn_index == len(questions) - 1 else "false",
                    # Which earlier turns' answers this turn's program actually consumes. Error
                    # propagation is measured against exactly these, not against mere adjacency —
                    # a wrong turn 1 only poisons turn 3 if turn 3 uses what turn 1 produced.
                    "reuses_turns": ",".join(str(t) for t in reused),
                    "reuses_prior_answer": "true" if reused else "false",
                },
            )

    def manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="convfinqa",
            official_source="czyssrs/ConvFinQA",
            paper_url="https://arxiv.org/abs/2210.03849",
            repository_url="https://github.com/czyssrs/ConvFinQA",
            version_or_commit=_PINNED_COMMIT,
            download_method="https (pinned commit, data.zip)",
            official_splits=("train", "dev", "test"),
            local_splits=("train", "dev"),
            license="MIT",
            redistribution_status="redistributable",
            expected_files=("dev.json", "train.json"),
            status=AdapterStatus.PARTIAL,
            status_tested_at="2026-07-11T00:00:00Z",
            known_limitations=(
                "The TEST split's gold answers were never publicly released — they exist only "
                "behind a CodaLab submission. It is therefore NOT supported, rather than faked "
                "with a locally re-derived split. Status is 'partial' for exactly this reason.",
                "Turns are NOT flattened into independent questions. Turn 1 of a real conversation "
                "is 'and what was it in 2005?', which means nothing without turn 0. Flattening is "
                "the tempting shortcut and it destroys the only thing this dataset measures.",
                "Two protocols, never mixed into one score: gold_history (each turn gets the GOLD "
                "prior conversation — isolates per-turn reasoning, and is the official setting) and "
                "model_history (each turn gets the model's OWN prior answers — the only way to "
                "measure error propagation). The gap between them is the interesting number.",
                "Under gold_history the prior turns' gold answers legitimately enter the prompt — "
                "that IS the protocol. The current turn's gold never does; the leakage suite has an "
                "explicit narrow exemption saying so.",
            ),
        )
