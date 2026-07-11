"""TAT-QA dataset adapter — the official train/dev/test-gold JSON mapped into ``CanonicalSample``.

Shape of a raw TAT-QA record (read from the official ``dataset_raw/*.json``, not from the paper):

.. code-block:: text

    {
      "table": {"uid": "...", "table": [["", "2019", "2018"], ["Fixed Price", "$ 1,452.4", ...]]},
      "paragraphs": [{"uid": "...", "order": 1, "text": "Sales by Contract Type: ..."}],
      "questions": [{
        "uid": "...", "order": 1, "question": "What is the amount of total sales in 2019?",
        "answer": ["$1,496.5"],          # a LIST for span/multi-span; a scalar for arithmetic/count
        "derivation": "1,452.4 + 44.1",
        "answer_type": "span" | "multi-span" | "arithmetic" | "count",
        "answer_from": "text" | "table" | "table-text",
        "scale": "" | "thousand" | "million" | "billion" | "percent",
        "req_comparison": false
      }]
    }

One passage carries several questions, so **one raw record fans out into several samples**, each
re-attaching the same table and paragraphs as its context.

``scale`` and ``answer_type`` are carried through in ``metadata`` because the official metric needs
both: the scale is folded into the compared string, and ``arithmetic``/``count`` answers have their
F1 forced equal to their EM (see ``evaluation/native/tatqa.py``).

**The test split's gold answers really are public** — ``dataset_raw/tatqa_dataset_test_gold.json``
is in the upstream repository. (They were held out originally; this was verified by reading the
repo, not assumed.)
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from financebench.datasets.base import DatasetAdapter, register_dataset
from financebench.datasets.downloader import download_file
from financebench.schemas.common import AnswerType, Scale, SplitOrigin
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

__all__ = ["TatQAAdapter"]

_PINNED_COMMIT = "870accc41953dcde885aabeb963d94aabdc0fbc3"
_RAW_BASE = f"https://raw.githubusercontent.com/NExTplusplus/TAT-QA/{_PINNED_COMMIT}/dataset_raw"
_OFFICIAL_URLS: dict[str, str] = {
    "train": f"{_RAW_BASE}/tatqa_dataset_train.json",
    "dev": f"{_RAW_BASE}/tatqa_dataset_dev.json",
    "test": f"{_RAW_BASE}/tatqa_dataset_test_gold.json",
}
_DEFAULT_DATA_DIR = Path("data/downloads/tatqa")

_ANSWER_TYPES: dict[str, AnswerType] = {
    "span": AnswerType.TEXT,
    "multi-span": AnswerType.TEXT,
    "arithmetic": AnswerType.NUMERIC,
    "count": AnswerType.NUMERIC,
}

_SCALES: dict[str, Scale | None] = {
    "": None,
    "thousand": Scale.THOUSAND,
    "million": Scale.MILLION,
    "billion": Scale.BILLION,
    # TAT-QA calls "percent" a scale; this platform keeps scale (magnitude) and unit separate, so
    # it becomes unit="percent" with no magnitude scaling. The raw value is preserved in metadata
    # for the official metric, which needs TAT-QA's own conflated notion.
    "percent": None,
}


@register_dataset("tatqa")
class TatQAAdapter(DatasetAdapter):
    name = "tatqa"

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR

    def prepare(self) -> None:
        for split, url in _OFFICIAL_URLS.items():
            dest = self._data_dir / f"{split}.json"
            if dest.is_file():
                continue
            download_file(url, dest)

    def load(self, split: str) -> Sequence[CanonicalSample]:
        if split not in _OFFICIAL_URLS:
            raise DatasetLoadError(
                f"tatqa has no split {split!r}; available: {sorted(_OFFICIAL_URLS)}"
            )
        path = self._data_dir / f"{split}.json"
        if not path.is_file():
            raise DatasetLoadError(
                f"tatqa {split} split not found at {path}. Run `financebench prepare tatqa` "
                "first (or point this adapter at a data_dir containing it)."
            )
        records = json.loads(path.read_text(encoding="utf-8"))
        return list(self._to_samples(records, split))

    def _to_samples(
        self, records: Sequence[dict[str, Any]], split: str
    ) -> Iterator[CanonicalSample]:
        for record in records:
            context = self._context(record)
            for question in record.get("questions", []):
                sample = self._to_sample(question, context, split)
                if sample is not None:
                    yield sample

    def _context(self, record: dict[str, Any]) -> SampleContext:
        raw_table = record.get("table") or {}
        rows = tuple(tuple(str(cell) for cell in row) for row in raw_table.get("table", []))
        tables = (
            (Table(table_id=str(raw_table.get("uid", "table")), rows=rows, header_rows=1),)
            if rows
            else ()
        )
        paragraphs = tuple(
            str(paragraph["text"])
            for paragraph in sorted(
                record.get("paragraphs", []), key=lambda p: int(p.get("order", 0))
            )
            if paragraph.get("text")
        )
        return SampleContext(text=paragraphs, tables=tables)

    def _to_sample(
        self, question: dict[str, Any], context: SampleContext, split: str
    ) -> CanonicalSample | None:
        uid = str(question.get("uid", "")).strip()
        text = str(question.get("question", "")).strip()
        if not uid or not text:
            return None

        raw_answer = question.get("answer")
        if raw_answer is None or raw_answer == "":
            return None
        # span/multi-span answers are lists; arithmetic/count are scalars.
        answers = (
            [str(a) for a in raw_answer] if isinstance(raw_answer, list) else [str(raw_answer)]
        )
        if not answers:
            return None

        answer_type_raw = str(question.get("answer_type", "span"))
        scale_raw = str(question.get("scale", "") or "")

        numeric_value: float | None = None
        if answer_type_raw in ("arithmetic", "count") and len(answers) == 1:
            try:
                numeric_value = float(str(answers[0]).replace(",", "").replace("$", "").strip())
            except ValueError:
                numeric_value = None

        return CanonicalSample(
            benchmark="tatqa",
            benchmark_version=f"official@{_PINNED_COMMIT[:8]}",
            split=split,
            split_origin=SplitOrigin.OFFICIAL,
            sample_id=f"tatqa:{split}:{uid}",
            task_family=f"tatqa_{answer_type_raw.replace('-', '_')}",
            capability_tags=("table_text", "calculation")
            if answer_type_raw in ("arithmetic", "count")
            else ("table_text", "evidence_grounding"),
            question=text,
            context=context,
            gold=GoldAnswer(
                answer=" | ".join(answers),
                answer_type=_ANSWER_TYPES.get(answer_type_raw, AnswerType.TEXT),
                numeric_value=numeric_value,
                unit="percent" if scale_raw == "percent" else None,
                scale=_SCALES.get(scale_raw),
                # The official metric compares against the answer *list*, so keep it intact.
                acceptable_answers=tuple(answers),
            ),
            evaluation=EvaluationSpec(),
            source=SourceInfo(
                license="MIT",
                url=f"https://github.com/NExTplusplus/TAT-QA/tree/{_PINNED_COMMIT}",
                redistributable=True,
            ),
            metadata={
                # The official evaluator needs TAT-QA's own conflated scale-and-unit notion verbatim, which
                # this platform otherwise splits apart — so it is preserved here rather than lost.
                "scale": scale_raw,
                "answer_type": answer_type_raw,
                "answer_from": str(question.get("answer_from", "")),
                "derivation": str(question.get("derivation", "")),
                "req_comparison": str(question.get("req_comparison", "")),
            },
        )

    def manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="tatqa",
            official_source="NExTplusplus/TAT-QA",
            paper_url="https://aclanthology.org/2021.acl-long.254/",
            repository_url="https://github.com/NExTplusplus/TAT-QA",
            version_or_commit=_PINNED_COMMIT,
            download_method="https (pinned commit)",
            official_splits=("train", "dev", "test"),
            local_splits=("train", "dev", "test"),
            license="MIT",
            redistribution_status="redistributable",
            expected_files=("train.json", "dev.json", "test.json"),
            status=AdapterStatus.FULLY_SUPPORTED,
            status_tested_at="2026-07-11T00:00:00Z",
            known_limitations=(
                "The scale that the official evaluator expects alongside the answer is INFERRED "
                "from the model's output here. TAT-QA's reference model (TagOp) has a dedicated "
                "scale-prediction head; a general LLM does not, so the scale has to be read back "
                "out of what it wrote. That inference is OURS, not official — the EM/F1 metric "
                "itself is parity-tested against the real evaluator. See "
                "docs/research/metric_parity.md.",
                "One passage carries several questions, so its table and paragraphs are repeated "
                "as context for each — token cost per sample is higher than the question implies.",
            ),
        )
