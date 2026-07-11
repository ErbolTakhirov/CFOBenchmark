"""Assembles a retriever for a run, and grades what it found afterwards.

This is the seam between the retrieval package (which knows about pages and BM25 and nothing about
benchmarks) and the orchestration layer (which knows about runs and nothing about PDFs).

Two retrieval settings are supported and reported separately, because they answer different
questions and averaging them would flatter whichever is easier:

- **open-corpus** — the retriever searches every page of every filing (~12,000 pages). This is the
  honest hard setting: the system must work out *which company's filing* as well as which page.
- **document-scoped** — the question already names the filing, so the corpus is narrowed to it and
  the job is to find the right **page** within a 160-page document. This is the setting a real
  deployment usually has, because a user asking about 3M's 2018 capex is not asking you to guess
  the company.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from financebench.prompts.profiles import RetrievedChunk
from financebench.retrieval.corpus import PageCorpus, build_corpus
from financebench.retrieval.embeddings import OllamaEmbedder, build_embeddings
from financebench.retrieval.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RetrievalResult,
    Retriever,
)
from financebench.schemas.sample import CanonicalSample

__all__ = ["RetrievalPipeline", "build_pipeline"]


@dataclass
class RetrievalPipeline:
    """Everything a retrieval-mode run needs, plus the record of what it did."""

    corpus: PageCorpus
    retriever: Retriever
    top_k: int
    document_scoped: bool
    #: sample_id -> what the retriever returned. Populated during the run, graded after it.
    results: dict[str, RetrievalResult]

    @property
    def fingerprint(self) -> str:
        return self.corpus.fingerprint

    def to_json(self) -> dict[str, object]:
        return {
            "retriever": self.retriever.name,
            "top_k": self.top_k,
            "document_scoped": self.document_scoped,
            "corpus_pages": len(self.corpus),
            "corpus_documents": len(self.corpus.documents),
            "index_fingerprint": self.fingerprint,
        }

    def retrieve_for(
        self, sample: CanonicalSample
    ) -> tuple[tuple[RetrievedChunk, ...], RetrievalResult]:
        """Retrieve for one sample.

        **Only the question text is used as the query.** Not the gold evidence, not the gold answer,
        not the justification — the retriever is given exactly what a user would type. In the
        document-scoped setting the document *name* is also used, because the question names it and
        a real system would know it; that is a stated part of the setting, not a peek at gold.
        """
        query = sample.question
        if self.document_scoped:
            doc = sample.metadata.get("doc_name", "")
            if doc:
                query = f"{doc.replace('_', ' ')} {query}"

        result = self.retriever.retrieve(query, top_k=self.top_k)
        self.results[sample.sample_id] = result
        return result.to_chunks(), result


def build_pipeline(
    samples: list[CanonicalSample],
    *,
    pdf_dir: str | Path,
    retriever_name: str = "bm25",
    top_k: int = 5,
    document_scoped: bool = False,
    embed_cache_dir: str | Path | None = None,
) -> RetrievalPipeline:
    """Build the corpus and retriever a run needs.

    The corpus covers every document any sample in the run refers to. In the open-corpus setting
    that is the whole ~12,000-page collection — which is the point: the retriever has to find one
    page in it.
    """
    documents = {
        sample.metadata.get("doc_name", "") for sample in samples if sample.metadata.get("doc_name")
    }
    corpus = build_corpus(pdf_dir, documents=documents or None)

    retriever: Retriever = BM25Retriever(corpus)
    if retriever_name in ("dense", "hybrid"):
        embedder = OllamaEmbedder()
        if not embedder.available():
            # Silently degrading to BM25 while still calling itself "dense" would be a lie in the
            # run artifacts. Say what actually happened.
            raise RuntimeError(
                f"retriever={retriever_name!r} needs the '{embedder.model}' embedding model, which "
                "Ollama does not have. Run: ollama pull nomic-embed-text — or use --retriever bm25."
            )
        cache = Path(embed_cache_dir or Path(pdf_dir).parent / "embed_cache")
        vectors = build_embeddings(corpus, embedder, cache_dir=cache)
        dense = DenseRetriever(corpus, vectors)
        dense.set_query_embedder(embedder.embed)
        retriever = (
            dense if retriever_name == "dense" else HybridRetriever(BM25Retriever(corpus), dense)
        )

    return RetrievalPipeline(
        corpus=corpus,
        retriever=retriever,
        top_k=top_k,
        document_scoped=document_scoped,
        results={},
    )
