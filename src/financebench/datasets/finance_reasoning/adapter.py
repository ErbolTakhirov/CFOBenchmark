"""FinanceReasoning dataset adapter — easy / medium / hard, from the official repository.

Shape of a raw record (read from the official ``data/FinanceReasoning/easy.json``):

.. code-block:: text

    {
      "question_id": "test-0",
      "level": "easy",
      "question": "What is the average INR-GBP exchange rate in FY 2019? ...",
      "context": "{\\"USD\\": {\\"Weightage (%)\\": 53.6, \\"FY 2019\\": 70.07, ...}, ...}",
      "ground_truth": 0.01,
      "python_solution": "irn_gbp_2019 = df[\\"GBP\\"][\\"FY 2019\\"]\\n\\nanswer = 1.0 / irn_gbp_2019",
      "difficulty": 0.693...,
      "source": "CodeTAT-QA-test-8",
      "statistics": {...}
    }

``context`` is a **JSON-encoded dataframe-shaped dict**, not prose — the gold ``python_solution``
indexes into it as ``df["GBP"]["FY 2019"]``. It is rendered into a readable table for the model
rather than dumped as raw JSON.

**Licence blocker.** The upstream repository ships **no LICENCE file and states no licence in its
README** (verified this session). The default in that situation is "all rights reserved", so this
adapter **downloads at runtime and never redistributes the data** — there is no vendored fixture,
and the e2e test skips (loudly) when the data has not been fetched. Recorded in
``docs/licenses.md`` and in this manifest's ``known_limitations``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from financebench.datasets.base import DatasetAdapter, register_dataset
from financebench.datasets.downloader import download_file
from financebench.schemas.common import AnswerType, SplitOrigin
from financebench.schemas.manifest import AdapterStatus, DatasetManifest
from financebench.schemas.sample import (
    CanonicalSample,
    EvaluationSpec,
    GoldAnswer,
    SampleContext,
    SourceInfo,
    Table,
)
from financebench.utils.errors import DatasetLoadError

__all__ = ["FinanceReasoningAdapter"]

_PINNED_COMMIT = "b0fe6455396f831955e4eb988472b4a563403bc5"
_RAW_BASE = (
    "https://raw.githubusercontent.com/BUPT-Reasoning-Lab/FinanceReasoning/"
    f"{_PINNED_COMMIT}/data/FinanceReasoning"
)
_LEVELS = ("easy", "medium", "hard")
_OFFICIAL_URLS = {level: f"{_RAW_BASE}/{level}.json" for level in _LEVELS}
_DEFAULT_DATA_DIR = Path("data/downloads/finance_reasoning")

#: The official relative tolerance, carried onto each sample so the grading rule travels with the
#: data rather than living only in the metric.
_RELATIVE_TOLERANCE = 0.002


def _render_context(raw: str) -> tuple[tuple[str, ...], tuple[Table, ...]]:
    """Turn the JSON-encoded dataframe into a table a model can actually read.

    Falls back to the raw string if it isn't the expected nested dict — better to show the model
    something than to silently drop its only context.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ((raw,), ()) if raw else ((), ())

    if not isinstance(data, dict) or not data:
        return ((raw,), ())

    # {column: {row: value}} -> a grid with the row labels down the left.
    columns = list(data)
    if not all(isinstance(value, dict) for value in data.values()):
        return ((json.dumps(data, indent=1),), ())

    row_labels: list[str] = []
    for column in columns:
        for row in data[column]:
            if row not in row_labels:
                row_labels.append(str(row))

    rows: list[tuple[str, ...]] = [("", *(str(c) for c in columns))]
    for label in row_labels:
        rows.append((label, *(str(data[column].get(label, "")) for column in columns)))
    return (), (Table(table_id="context", rows=tuple(rows), header_rows=1),)


@register_dataset("finance_reasoning")
class FinanceReasoningAdapter(DatasetAdapter):
    name = "finance_reasoning"

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR

    def prepare(self) -> None:
        for level, url in _OFFICIAL_URLS.items():
            dest = self._data_dir / f"{level}.json"
            if dest.is_file():
                continue
            download_file(url, dest)

    def load(self, split: str) -> Sequence[CanonicalSample]:
        if split not in _LEVELS:
            raise DatasetLoadError(
                f"finance_reasoning has no split {split!r}; available: {list(_LEVELS)}"
            )
        path = self._data_dir / f"{split}.json"
        if not path.is_file():
            raise DatasetLoadError(
                f"finance_reasoning {split} split not found at {path}. Run "
                "`financebench prepare finance_reasoning` first. (The data is NOT bundled: "
                "upstream ships no licence, so it cannot be redistributed.)"
            )
        records = json.loads(path.read_text(encoding="utf-8"))
        return list(self._to_samples(records, split))

    def _to_samples(
        self, records: Sequence[dict[str, Any]], split: str
    ) -> Iterator[CanonicalSample]:
        for record in records:
            sample = self._to_sample(record, split)
            if sample is not None:
                yield sample

    def _to_sample(self, record: dict[str, Any], split: str) -> CanonicalSample | None:
        question_id = str(record.get("question_id", "")).strip()
        question = str(record.get("question", "")).strip()
        if not question_id or not question:
            return None
        if "ground_truth" not in record:
            return None

        gold_raw = record["ground_truth"]
        is_bool = isinstance(gold_raw, bool)
        numeric_value: float | None = None
        if not is_bool:
            try:
                numeric_value = float(gold_raw)
            except (TypeError, ValueError):
                return None

        text, tables = _render_context(str(record.get("context", "")))
        # 6 of the 1,000 easy records ship an EMPTY context upstream: the question is unanswerable
        # because there is nothing to answer it from. They are kept (dropping them would break
        # count-parity with published FinanceReasoning numbers, which include them) but flagged, so
        # a reader can see that a floor of ~0.6% is baked into the benchmark and is not the model's
        # fault. Surfaced in the coverage report and queryable via metadata.
        context_is_empty = not text and not tables

        return CanonicalSample(
            benchmark="finance_reasoning",
            benchmark_version=f"official@{_PINNED_COMMIT[:8]}",
            split=split,
            split_origin=SplitOrigin.OFFICIAL,
            sample_id=f"finance_reasoning:{split}:{question_id}",
            task_family=f"finance_reasoning_{split}",
            capability_tags=("calculation", "table_text"),
            question=question,
            context=SampleContext(text=text, tables=tables),
            gold=GoldAnswer(
                answer=str(gold_raw),
                answer_type=AnswerType.BOOLEAN if is_bool else AnswerType.NUMERIC,
                numeric_value=numeric_value,
                # The gold Python solution. Never shown to the model — see
                # tests/security/test_gold_answer_leakage.py.
                program=str(record.get("python_solution", "")) or None,
            ),
            evaluation=EvaluationSpec(relative_tolerance=_RELATIVE_TOLERANCE),
            source=SourceInfo(
                license="UNLICENSED (no LICENSE file upstream)",
                url=f"https://github.com/BUPT-Reasoning-Lab/FinanceReasoning/tree/{_PINNED_COMMIT}",
                redistributable=False,
            ),
            metadata={
                "ground_truth": str(gold_raw),
                "level": split,
                "difficulty": str(record.get("difficulty", "")),
                "source_dataset": str(record.get("source", "")),
                "context_empty": "true" if context_is_empty else "false",
            },
        )

    def manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="finance_reasoning",
            official_source="BUPT-Reasoning-Lab/FinanceReasoning",
            paper_url="https://arxiv.org/abs/2506.05828",
            repository_url="https://github.com/BUPT-Reasoning-Lab/FinanceReasoning",
            version_or_commit=_PINNED_COMMIT,
            download_method="https (pinned commit)",
            official_splits=_LEVELS,
            local_splits=_LEVELS,
            license="UNLICENSED (no LICENSE file, no licence statement in the README)",
            redistribution_status="not_redistributable",
            expected_files=("easy.json", "medium.json", "hard.json"),
            status=AdapterStatus.FULLY_SUPPORTED,
            status_tested_at="2026-07-11T00:00:00Z",
            known_limitations=(
                "LICENCE BLOCKER: upstream ships no LICENSE file and states no licence in its "
                "README (verified 2026-07-11). The default is therefore 'all rights reserved'. "
                "This adapter downloads at runtime and does NOT redistribute the data — there is "
                "no vendored fixture, and the e2e test skips when the data has not been fetched.",
                "The official normalize() ends in a bare eval() on model output. That is not "
                "ported: it is arbitrary code execution on text a model produced. A safe AST "
                "arithmetic evaluator is used instead, and the parity suite asserts the two agree "
                "on realistic predictions. See docs/research/metric_parity.md.",
                "Grading uses the official 0.2% RELATIVE tolerance (eps = |gt| * 0.002). Because "
                "it is relative to the gold, a gold of exactly 0 admits only an exact 0.",
                "6 of the 1,000 'easy' records ship an EMPTY context upstream — the question "
                "cannot be answered because there is nothing to answer it from. They are kept "
                "(dropping them would break count-parity with published numbers, which include "
                "them) and flagged with metadata.context_empty=true. No model can score on them, "
                "so ~0.6% of the easy split is an unavoidable floor that is not the model's fault.",
            ),
        )
