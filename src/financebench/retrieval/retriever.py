"""The retriever contract, and the guarantee that it never sees the answer.

A retrieval benchmark that lets the retriever peek at the gold evidence measures nothing. It is the
same failure as gold leakage in a prompt, and it is *easier* to commit accidentally, because the
gold evidence is right there on the sample and using it would "work".

So the contract is narrow on purpose: a retriever is handed **a query string and a corpus**, and
nothing else. It never receives a :class:`CanonicalSample`. It cannot reach `sample.gold` because
it is never given a sample to reach it from.

``tests/security/test_retrieval_leakage.py`` pins this down the same way the prompt-leakage suite
does: scrub a sample's gold evidence to sentinels, retrieve again, and assert the retrieved pages
are identical. If retrieval cannot change when the gold changes, the gold is not being used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from financebench.prompts.profiles import RetrievedChunk
from financebench.retrieval.bm25 import BM25Index, RetrievedPage
from financebench.retrieval.corpus import PageCorpus

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "RetrievalResult",
    "Retriever",
]


@dataclass(frozen=True)
class RetrievalResult:
    """What a retriever found, and enough provenance to grade it afterwards."""

    pages: tuple[RetrievedPage, ...]
    retriever: str
    top_k: int

    def to_chunks(self) -> tuple[RetrievedChunk, ...]:
        """The prompt-facing view: text plus where it came from, so the model can cite it."""
        return tuple(
            RetrievedChunk(
                document_id=hit.page.document_id,
                text=hit.page.text[:6000],
                page=hit.page.page,
                score=hit.score,
            )
            for hit in self.pages
        )

    @property
    def documents(self) -> set[str]:
        return {hit.page.document_id for hit in self.pages}

    @property
    def chunk_ids(self) -> list[str]:
        return [hit.page.chunk_id for hit in self.pages]


class Retriever(ABC):
    """Given a *query string* and nothing else, find pages.

    The signature is the guarantee. There is no sample here, so there is no gold here.
    """

    name: str = ""

    @abstractmethod
    def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult: ...


class BM25Retriever(Retriever):
    """Lexical. Strong on financial documents precisely because numbers and line-item names are
    rare, discriminating tokens — which is exactly what BM25's IDF rewards."""

    name = "bm25"

    def __init__(self, corpus: PageCorpus) -> None:
        self._index = BM25Index(corpus)

    def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult:
        return RetrievalResult(
            pages=tuple(self._index.search(query, top_k=top_k)),
            retriever=self.name,
            top_k=top_k,
        )


class DenseRetriever(Retriever):
    """Semantic, via ``nomic-embed-text`` on the local Ollama.

    Embeddings are computed once per page and cached, because embedding ~12,000 pages is the
    expensive part and doing it per-query would make the benchmark unusable.
    """

    name = "dense"

    def __init__(self, corpus: PageCorpus, embeddings: dict[str, list[float]]) -> None:
        self._corpus = corpus
        self._embeddings = embeddings
        self._query_embed: object | None = None

    def set_query_embedder(self, embed: object) -> None:
        self._query_embed = embed

    def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult:
        embed = self._query_embed
        if embed is None or not self._embeddings:
            return RetrievalResult(pages=(), retriever=self.name, top_k=top_k)

        query_vector = embed(query)  # type: ignore[operator]
        scored: list[tuple[float, str]] = []
        for chunk_id, vector in self._embeddings.items():
            scored.append((_cosine(query_vector, vector), chunk_id))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))

        hits = []
        for rank, (score, chunk_id) in enumerate(scored[:top_k], start=1):
            page = self._corpus.get(chunk_id)
            if page is not None:
                hits.append(RetrievedPage(page=page, score=score, rank=rank))
        return RetrievalResult(pages=tuple(hits), retriever=self.name, top_k=top_k)


class HybridRetriever(Retriever):
    """BM25 + dense, fused by reciprocal rank.

    RRF rather than score-mixing, because BM25 scores and cosine similarities are not on the same
    scale and normalizing them into agreement is an invitation to tune a constant until the number
    looks good. Rank is the one thing both retrievers agree on the meaning of.
    """

    name = "hybrid"

    def __init__(self, lexical: Retriever, dense: Retriever, *, k: int = 60) -> None:
        self._lexical = lexical
        self._dense = dense
        self._k = k

    def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult:
        fused: dict[str, float] = {}
        pages: dict[str, RetrievedPage] = {}
        for retriever in (self._lexical, self._dense):
            result = retriever.retrieve(query, top_k=top_k * 2)
            for hit in result.pages:
                fused[hit.page.chunk_id] = fused.get(hit.page.chunk_id, 0.0) + 1.0 / (
                    self._k + hit.rank
                )
                pages[hit.page.chunk_id] = hit

        ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
        hits = [
            RetrievedPage(page=pages[chunk_id].page, score=score, rank=rank)
            for rank, (chunk_id, score) in enumerate(ordered[:top_k], start=1)
        ]
        return RetrievalResult(pages=tuple(hits), retriever=self.name, top_k=top_k)


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
