"""Dense embeddings via ``nomic-embed-text`` on the local Ollama.

Embedding ~12,000 pages is the expensive part of dense retrieval — minutes, not milliseconds — so
vectors are computed once and cached to disk under the corpus fingerprint. Re-running a benchmark
must not re-embed a corpus that has not changed.

This is optional. BM25 alone is a complete, honest retrieval baseline (and on financial documents a
strong one — line-item names and figures are exactly the rare, discriminating tokens BM25's IDF
rewards). If the embedding model is not available, dense and hybrid retrieval are simply not
offered, rather than silently degrading into something that looks like they ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from financebench.retrieval.corpus import PageCorpus

__all__ = ["OllamaEmbedder", "build_embeddings"]

_DEFAULT_MODEL = "nomic-embed-text"
_DEFAULT_BASE_URL = "http://localhost:11434"

#: Embedding models have their own context limit, and a 10-K page can exceed it. Truncating is
#: honest — the alternative is a failed request — and the head of a page is where its headings and
#: line-item labels live, which is what the query is trying to match.
_MAX_CHARS = 6000


class OllamaEmbedder:
    """Text -> vector, via Ollama's native embeddings endpoint."""

    def __init__(
        self, model: str = _DEFAULT_MODEL, base_url: str = _DEFAULT_BASE_URL, timeout: float = 60.0
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def available(self) -> bool:
        try:
            response = self._client.get(f"{self._base_url}/api/tags", timeout=5.0)
            names = {m["name"].split(":")[0] for m in response.json().get("models", [])}
            return self.model.split(":")[0] in names
        except (httpx.HTTPError, KeyError, ValueError):
            return False

    def embed(self, text: str) -> list[float]:
        response = self._client.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self.model, "prompt": text[:_MAX_CHARS]},
        )
        response.raise_for_status()
        vector = response.json().get("embedding") or []
        return [float(x) for x in vector]

    def close(self) -> None:
        self._client.close()


def build_embeddings(
    corpus: PageCorpus,
    embedder: OllamaEmbedder,
    *,
    cache_dir: str | Path,
    progress_every: int = 500,
) -> dict[str, list[float]]:
    """Embed every page in ``corpus``, cached under the corpus fingerprint.

    The cache key is the corpus fingerprint, so a changed corpus gets fresh vectors rather than
    silently reusing embeddings of pages that no longer exist.
    """
    cache_path = Path(cache_dir) / f"embeddings.{embedder.model}.{corpus.fingerprint}.json"
    if cache_path.is_file():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return {k: [float(x) for x in v] for k, v in raw.items()}

    vectors: dict[str, list[float]] = {}
    for index, page in enumerate(corpus.pages, start=1):
        if not page.text.strip():
            continue
        try:
            vectors[page.chunk_id] = embedder.embed(page.text)
        except httpx.HTTPError:
            # One page that fails to embed is one page the dense retriever cannot see. That is a
            # coverage gap, not a reason to abandon a 12,000-page corpus — and it is visible,
            # because the vector count won't match the page count.
            continue
        if progress_every and index % progress_every == 0:
            print(f"  embedded {index}/{len(corpus)} pages", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(vectors), encoding="utf-8")
    return vectors
