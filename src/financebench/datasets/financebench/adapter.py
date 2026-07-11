"""FinanceBench (Patronus AI) — the **public 150-question subset**, in two evaluation modes.

Status is `supported_public_subset` and stays that way. The full FinanceBench is 10,231 questions
and is gated behind a direct request to Patronus; this is the 150 open rows, and it is never
described as anything else.

**FinanceBench ships no evaluator.** The upstream repo has an `evaluation_playground.ipynb` and
nothing more. So every metric this project reports on it is *ours*, is named `financebench_*`, and
must never be compared against a published FinanceBench number.

Shape of a raw row (read from the real `financebench_open_source.jsonl`):

.. code-block:: text

    {
      "financebench_id": "financebench_id_03029",
      "company": "3M", "doc_name": "3M_2018_10K",
      "question_type": "metrics-generated" | "domain-relevant" | "novel-generated",
      "question_reasoning": "Information extraction" | "Logical reasoning ..." | null,
      "question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M? ...",
      "answer": "$1577.00",
      "justification": "The metric capital expenditures was directly extracted from ...",
      "evidence": [{"evidence_text": "...", "doc_name": "...",
                    "evidence_page_num": "59", "evidence_text_full_page": "..."}]
    }

The gold answers come in three shapes, and pretending otherwise is how a benchmark ends up
measuring formatting:

======================  =====  ===================================================================
Shape                   Count  How it is scored
======================  =====  ===================================================================
numeric (``$1577.00``)  52     deterministically, with the numeric parser
boolean (``Yes/No...``) 37     deterministically, on the leading yes/no
free-text analysis      61     **cannot** be exact-matched. Scored by an optional judge; reported
                               as ``not_evaluated`` — never as zero — when no judge is configured
======================  =====  ===================================================================

The **unsupported-numeric-claim** metric, by contrast, applies to all 150 and is fully
deterministic: a number the model states that appears nowhere in the evidence it was given is a
hallucination, whatever shape the gold answer takes. That is the single most important number this
benchmark produces.

Two modes:

- ``context_given`` — the official evidence text is handed to the model. Measures the *model*.
- ``retrieval_required`` — the model gets nothing but a corpus of PDF pages and must find its own
  evidence. Measures the *retrieval system*. The retriever never sees gold (``retrieval/``).
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
    DocumentRef,
    EvaluationSpec,
    Evidence,
    GoldAnswer,
    SampleContext,
    SourceInfo,
)
from financebench.utils.errors import DatasetLoadError

__all__ = ["FinanceBenchAdapter", "answer_shape"]

_PINNED_COMMIT = "cc39aeb4afdf33909ee1412188bf89035950c2eb"
_RAW_BASE = f"https://raw.githubusercontent.com/patronus-ai/financebench/{_PINNED_COMMIT}"
_OPEN_SOURCE_URL = f"{_RAW_BASE}/data/financebench_open_source.jsonl"
_DEFAULT_DATA_DIR = Path("data/downloads/financebench")

#: The three gold-answer shapes. See the module docstring — conflating them is how a benchmark
#: ends up scoring formatting instead of finance.
NUMERIC = "numeric"
BOOLEAN = "boolean"
ANALYTICAL = "analytical"


def answer_shape(answer: str) -> str:
    """Classify a gold answer so it can be scored by something that actually applies to it."""
    import re

    from financebench.evaluation.numeric import parse_numeric_answer

    text = answer.strip()
    if re.match(r"^\s*(yes|no)\b", text, re.IGNORECASE):
        return BOOLEAN
    # A short answer that parses as a number is a numeric answer. A long one that happens to start
    # with a figure is prose, and treating it as numeric would score an essay by its first digit.
    if len(text) < 40 and parse_numeric_answer(text) is not None:
        return NUMERIC
    return ANALYTICAL


@register_dataset("financebench")
class FinanceBenchAdapter(DatasetAdapter):
    name = "financebench"

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR

    def prepare(self) -> None:
        """Fetch the 150-row open subset, and the PDFs the retrieval mode needs.

        The PDFs are ~150 MB across 84 documents and are only required for
        ``retrieval_required``; the question set alone is a few hundred KB. Both are fetched here so
        that `prepare` means prepared.
        """
        dest = self._data_dir / "financebench_open_source.jsonl"
        if not dest.is_file():
            download_file(_OPEN_SOURCE_URL, dest)

        pdf_dir = self._data_dir / "pdfs"
        for doc_name in sorted(self._document_names()):
            pdf = pdf_dir / f"{doc_name}.pdf"
            if pdf.is_file() and pdf.stat().st_size > 0:
                continue
            try:
                download_file(f"{_RAW_BASE}/pdfs/{doc_name}.pdf", pdf, max_bytes=100 * 1024 * 1024)
            except Exception:  # a single missing PDF must not sink the whole prepare
                continue

    def _rows(self) -> list[dict[str, Any]]:
        path = self._data_dir / "financebench_open_source.jsonl"
        if not path.is_file():
            raise DatasetLoadError(
                f"financebench open subset not found at {path}. "
                "Run `financebench prepare financebench` first."
            )
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _document_names(self) -> set[str]:
        return {str(row["doc_name"]) for row in self._rows()}

    def load(self, split: str) -> Sequence[CanonicalSample]:
        if split != "open_source":
            raise DatasetLoadError(
                f"financebench has no split {split!r}; available: ['open_source']. "
                "(The full 10,231-question set is gated behind a request to Patronus AI and is "
                "not supported here.)"
            )
        return list(self._to_samples(self._rows(), split))

    def _to_samples(self, rows: Sequence[dict[str, Any]], split: str) -> Iterator[CanonicalSample]:
        for row in rows:
            sample = self._to_sample(row, split)
            if sample is not None:
                yield sample

    def _to_sample(self, row: dict[str, Any], split: str) -> CanonicalSample | None:
        row_id = str(row.get("financebench_id", "")).strip()
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not row_id or not question or not answer:
            return None

        raw_evidence = row.get("evidence") or []
        shape = answer_shape(answer)

        # The model-facing context is the evidence TEXT. The doc name, page number and
        # justification stay on the evaluator's side of the fence — they are how we grade the
        # answer, not part of the question.
        context_blocks = tuple(
            str(block.get("evidence_text", "")).strip()
            for block in raw_evidence
            if str(block.get("evidence_text", "")).strip()
        )

        evidence = tuple(
            Evidence(
                document_id=str(block.get("doc_name", "")) or None,
                page=_page(block.get("evidence_page_num")),
                text_snippet=str(block.get("evidence_text", ""))[:4000] or None,
            )
            for block in raw_evidence
        )

        numeric_value: float | None = None
        if shape == NUMERIC:
            from financebench.evaluation.numeric import parse_numeric_answer

            parsed = parse_numeric_answer(answer)
            numeric_value = parsed.resolved_value if parsed is not None else None

        answer_type = {
            NUMERIC: AnswerType.NUMERIC,
            BOOLEAN: AnswerType.BOOLEAN,
            ANALYTICAL: AnswerType.TEXT,
        }[shape]

        doc_name = str(row.get("doc_name", ""))
        return CanonicalSample(
            benchmark="financebench",
            benchmark_version=f"open_source@{_PINNED_COMMIT[:8]}",
            split=split,
            split_origin=SplitOrigin.PUBLIC_SUBSET,
            sample_id=f"financebench:{split}:{row_id}",
            task_family=f"financebench_{shape}",
            capability_tags=(
                ("evidence_grounding", "calculation")
                if shape == NUMERIC
                else ("evidence_grounding", "analysis")
            ),
            question=question,
            context=SampleContext(
                text=context_blocks,
                documents=(DocumentRef(document_id=doc_name, title=doc_name),) if doc_name else (),
            ),
            gold=GoldAnswer(
                answer=answer,
                answer_type=answer_type,
                numeric_value=numeric_value,
                evidence=evidence,
            ),
            evaluation=EvaluationSpec(requires_citation=True),
            source=SourceInfo(
                license="CC BY-NC-4.0",
                url="https://github.com/patronus-ai/financebench",
                redistributable=False,
            ),
            metadata={
                "company": str(row.get("company", "")),
                "doc_name": doc_name,
                "question_type": str(row.get("question_type", "")),
                "question_reasoning": str(row.get("question_reasoning") or ""),
                "answer_shape": shape,
                # Evaluator-only. Never rendered into a prompt.
                "justification": str(row.get("justification", ""))[:2000],
                "gold_pages": ",".join(
                    str(_page(b.get("evidence_page_num")) or "") for b in raw_evidence
                ),
            },
        )

    def manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="financebench",
            official_source="patronus-ai/financebench",
            paper_url="https://arxiv.org/abs/2311.11944",
            repository_url="https://github.com/patronus-ai/financebench",
            version_or_commit=_PINNED_COMMIT,
            download_method="https (pinned commit)",
            official_splits=("open_source",),
            local_splits=("open_source",),
            license="CC BY-NC-4.0",
            redistribution_status="not_redistributable (non-commercial)",
            expected_files=("financebench_open_source.jsonl", "pdfs/"),
            status=AdapterStatus.SUPPORTED_PUBLIC_SUBSET,
            status_tested_at="2026-07-11T00:00:00Z",
            known_limitations=(
                "PUBLIC SUBSET ONLY: 150 of the 10,231 FinanceBench questions. The full set is "
                "gated behind a direct request to Patronus AI. This is never described as the "
                "full FinanceBench.",
                "FinanceBench ships NO EVALUATOR — the upstream repo has an evaluation notebook "
                "and nothing more. Every metric reported here is OURS, is named financebench_*, "
                "and must not be compared to any published FinanceBench number.",
                "Gold answers come in three shapes: 52 numeric, 37 boolean, 61 free-text "
                "analysis. The 61 analytical answers CANNOT be exact-matched; they are scored by "
                "an optional LLM judge and reported as not_evaluated — never as zero — when no "
                "judge is configured.",
                "The unsupported-numeric-claim metric applies to all 150 and is fully "
                "deterministic: a number the model states that appears nowhere in its evidence is "
                "a hallucination regardless of the gold answer's shape.",
            ),
        )


def _page(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None
