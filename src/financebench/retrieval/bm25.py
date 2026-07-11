"""BM25 — hand-rolled, deterministic, no dependency.

`rank_bm25` would have done, but this is ~60 lines of well-specified arithmetic and taking a
dependency for it buys nothing. Hand-rolling also means the tokenizer is *ours* and can be made to
care about the things financial retrieval actually turns on:

- **numbers are kept as tokens.** A query asking for "FY2018 capital expenditure" is best served by
  a page containing "2018", and a tokenizer that strips digits — as many do — throws away the single
  most discriminating feature in a financial document.
- **numeric punctuation is normalized** (`1,577.00` → `1577.00`), so a query and a table cell agree
  on what the number is.

Deterministic by construction: no randomness, no learned state, and ties broken by chunk id so the
same query over the same corpus always returns the same pages in the same order. A retriever whose
output wobbles between runs makes every downstream score unreproducible.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from financebench.retrieval.corpus import Page, PageCorpus

__all__ = ["BM25Index", "RetrievedPage", "tokenize"]

_K1 = 1.5
_B = 0.75

_TOKEN_RE = re.compile(r"[a-z]+|\d[\d,\.]*")

#: Words so common in a 10-K that they discriminate nothing. Kept deliberately short: BM25's IDF
#: already handles the general case, and an aggressive stop-list would drop "net" and "income".
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "at", "by", "is", "are",
        "was", "were", "be", "been", "as", "that", "this", "it", "its", "with", "from", "we",
    }
)  # fmt: skip


def tokenize(text: str) -> list[str]:
    """Lowercase word and number tokens. Numbers survive — they are the point."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        token = raw.rstrip(".,").replace(",", "") if raw[0].isdigit() else raw
        if not token or token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


class RetrievedPage:
    """A page a retriever returned, with the score that got it there."""

    __slots__ = ("page", "rank", "score")

    def __init__(self, page: Page, score: float, rank: int) -> None:
        self.page = page
        self.score = score
        self.rank = rank


class BM25Index:
    """Okapi BM25 over a page corpus."""

    def __init__(self, corpus: PageCorpus, *, k1: float = _K1, b: float = _B) -> None:
        self._corpus = corpus
        self._k1 = k1
        self._b = b

        self._doc_tokens: list[list[str]] = [tokenize(page.text) for page in corpus.pages]
        self._doc_len = [len(tokens) for tokens in self._doc_tokens]
        self._avg_len = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0

        self._term_freqs: list[Counter[str]] = [Counter(tokens) for tokens in self._doc_tokens]
        document_freq: Counter[str] = Counter()
        for tokens in self._doc_tokens:
            document_freq.update(set(tokens))

        n = len(self._doc_tokens)
        # Robertson/Sparck-Jones IDF, with the +1 that keeps it non-negative for terms appearing in
        # more than half the corpus. Without it, a term in most documents gets a *negative* weight
        # and a page can be penalised for containing the word you searched for.
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in document_freq.items()
        }

    @property
    def corpus(self) -> PageCorpus:
        return self._corpus

    def search(self, query: str, top_k: int = 5) -> list[RetrievedPage]:
        """The ``top_k`` best pages for ``query``. Deterministic, including its tie-breaks."""
        query_terms = tokenize(query)
        if not query_terms or not self._doc_tokens:
            return []

        scores: list[tuple[float, int]] = []
        for index, freqs in enumerate(self._term_freqs):
            length = self._doc_len[index]
            if length == 0:
                continue
            score = 0.0
            for term in query_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                denominator = tf + self._k1 * (
                    1.0 - self._b + self._b * (length / (self._avg_len or 1.0))
                )
                score += idf * (tf * (self._k1 + 1.0)) / denominator
            if score > 0:
                scores.append((score, index))

        # Sort by score desc, then by chunk id — so equal-scoring pages come back in a stable order
        # rather than whatever order the corpus happened to be built in.
        scores.sort(key=lambda pair: (-pair[0], self._corpus.pages[pair[1]].chunk_id))
        return [
            RetrievedPage(page=self._corpus.pages[index], score=score, rank=rank)
            for rank, (score, index) in enumerate(scores[:top_k], start=1)
        ]
