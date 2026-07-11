"""Retrieval metrics, and the attribution that makes them actionable.

A RAG system that answers 30 % of questions has two very different diseases, and the treatments are
opposite:

- **the retriever never found the page** — improve retrieval; the model was never given a chance;
- **the retriever found the page and the model still got it wrong** — improve the model; more
  retrieval will not help.

A single "RAG accuracy" number cannot tell these apart, so it sends you off to fix the wrong
component. Every metric here exists to keep them separate.

The headline number is the **retrieval loss**: `context_given` accuracy minus `retrieval_required`
accuracy on the *same questions*. It is the price the system pays for having to find its own
evidence, and it is the number a person deciding whether to build a RAG pipeline actually needs.
"""

from __future__ import annotations

from dataclasses import dataclass

from financebench.evaluation.failures import FailureType
from financebench.retrieval.retriever import RetrievalResult
from financebench.schemas.sample import CanonicalSample

__all__ = [
    "RetrievalScore",
    "attribute_retrieval_failure",
    "score_retrieval",
]


@dataclass(frozen=True)
class RetrievalScore:
    """How well the retriever did on one question — judged *after* inference, against the gold."""

    document_hit: bool
    page_hit: bool
    evidence_precision: float
    evidence_recall: float
    n_gold_pages: int
    n_retrieved: int
    gold_page_rank: int | None  # where the right page landed; None if never retrieved

    @property
    def evidence_f1(self) -> float:
        p, r = self.evidence_precision, self.evidence_recall
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def _gold_pages(sample: CanonicalSample) -> set[str]:
    """The pages the answer actually lives on — `document#pN`, evaluator-side only."""
    return {
        f"{e.document_id}#p{e.page}"
        for e in sample.gold.evidence
        if e.document_id and e.page is not None
    }


def score_retrieval(sample: CanonicalSample, result: RetrievalResult) -> RetrievalScore:
    """Grade a retrieval **after** the fact. Gold is used here and nowhere earlier."""
    gold = _gold_pages(sample)
    gold_docs = {e.document_id for e in sample.gold.evidence if e.document_id}
    retrieved = set(result.chunk_ids)

    hit_pages = gold & retrieved
    precision = (len(hit_pages) / len(retrieved)) if retrieved else 0.0
    recall = (len(hit_pages) / len(gold)) if gold else 0.0

    rank: int | None = None
    for hit in result.pages:
        if hit.page.chunk_id in gold:
            rank = hit.rank
            break

    return RetrievalScore(
        document_hit=bool(gold_docs & result.documents),
        page_hit=bool(hit_pages),
        evidence_precision=precision,
        evidence_recall=recall,
        n_gold_pages=len(gold),
        n_retrieved=len(retrieved),
        gold_page_rank=rank,
    )


def attribute_retrieval_failure(
    retrieval: RetrievalScore, *, answer_correct: bool, refused: bool
) -> FailureType | None:
    """Whose fault was it — the retriever's, or the model's?

    This is the whole point of the mode. Returns ``None`` when nothing went wrong.
    """
    if answer_correct:
        return None

    if not retrieval.document_hit:
        # It didn't even find the right filing. Nothing the model did after that mattered.
        return FailureType.WRONG_DOCUMENT

    if not retrieval.page_hit:
        # Right company, wrong page. The evidence was never in front of the model.
        return FailureType.RETRIEVAL_MISS

    # The right page WAS retrieved and the model still didn't get there. More retrieval will not
    # fix this — the evidence was on the screen.
    if refused:
        return FailureType.UNNECESSARY_REFUSAL
    return FailureType.GENERATION_ERROR_AFTER_RETRIEVAL
