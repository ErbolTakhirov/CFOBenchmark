"""Real document retrieval over real financial PDFs.

The mode this package exists for — ``retrieval_required`` — measures a *retrieval system*, not a
model. In ``context_given`` the model is handed the evidence and asked to reason. Here it is handed
a corpus of ~12,000 pages of 10-K filings and must find the evidence itself, which is the job an
actual financial RAG deployment does.

The two are never averaged into one number. The difference between them — the **retrieval loss** —
is itself the most useful thing this package produces.

**The retriever never sees gold.** Its whole interface is `retrieve(query) -> pages`; it is never
handed a sample, so it has nothing to peek at. Enforced by `tests/security/test_retrieval_leakage.py`.
"""

from __future__ import annotations

from financebench.retrieval.bm25 import BM25Index, tokenize
from financebench.retrieval.corpus import Page, PageCorpus, build_corpus
from financebench.retrieval.embeddings import OllamaEmbedder, build_embeddings
from financebench.retrieval.metrics import (
    RetrievalScore,
    attribute_retrieval_failure,
    score_retrieval,
)
from financebench.retrieval.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RetrievalResult,
    Retriever,
)

__all__ = [
    "BM25Index",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "OllamaEmbedder",
    "Page",
    "PageCorpus",
    "RetrievalResult",
    "RetrievalScore",
    "Retriever",
    "attribute_retrieval_failure",
    "build_corpus",
    "build_embeddings",
    "score_retrieval",
    "tokenize",
]
