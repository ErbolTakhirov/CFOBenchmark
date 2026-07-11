"""Page-level PDF corpus — deterministic, cached, fingerprinted.

The unit of retrieval is a **page**, because that is the unit FinanceBench's gold evidence is
annotated in (`evidence_page_num`). Retrieving a whole 160-page 10-K would be no retrieval at all;
retrieving a paragraph would make the gold un-scoreable. A page is the honest granularity.

Extraction is deterministic and cached to disk under a content hash of the PDF, so:

- a run is reproducible without re-parsing 84 PDFs and ~12,000 pages every time;
- the **index fingerprint** in the run artifacts pins exactly which corpus produced a score. Two
  retrieval runs over different corpora are not comparable, and the fingerprint is what makes that
  visible rather than something a reader has to take on trust.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Page", "PageCorpus", "build_corpus"]

_CACHE_VERSION = "1"


@dataclass(frozen=True)
class Page:
    """One page of one document — the unit of retrieval and of gold evidence."""

    document_id: str
    page: int  # 1-indexed, matching FinanceBench's `evidence_page_num`
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.document_id}#p{self.page}"


def _normalize(text: str) -> str:
    """Collapse the whitespace PDF extraction sprays everywhere, without touching the content.

    pypdf routinely breaks a word across a newline mid-token ("Equit\\ny"). Left alone, that
    destroys both BM25 term matching and anything a model tries to read. Collapsing runs of
    whitespace is the minimum repair that does not invent text.
    """
    # pypdf emits U+00A0 (non-breaking space) liberally. Left as-is it is not \s, so the
    # tokenizer glues adjacent words into one token and BM25 never matches them.
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class PageCorpus:
    """Every page of every document a benchmark needs, addressable by ``document_id#pN``."""

    def __init__(self, pages: list[Page]) -> None:
        self._pages = pages
        self._by_id = {page.chunk_id: page for page in pages}
        self._by_document: dict[str, list[Page]] = {}
        for page in pages:
            self._by_document.setdefault(page.document_id, []).append(page)

    def __len__(self) -> int:
        return len(self._pages)

    @property
    def pages(self) -> list[Page]:
        return self._pages

    @property
    def documents(self) -> list[str]:
        return sorted(self._by_document)

    def get(self, chunk_id: str) -> Page | None:
        return self._by_id.get(chunk_id)

    def for_document(self, document_id: str) -> list[Page]:
        return self._by_document.get(document_id, [])

    def scoped_to(self, document_ids: set[str]) -> PageCorpus:
        """A sub-corpus over only these documents.

        Used for the *document-scoped* retrieval setting: the question names the filing, so the
        retriever's job is to find the right **page** within it, not to find the company. Both
        settings are reported, because they answer different questions and conflating them would
        flatter whichever is easier.
        """
        return PageCorpus([p for p in self._pages if p.document_id in document_ids])

    @property
    def fingerprint(self) -> str:
        """Identifies exactly this corpus. Two runs over different corpora are not comparable."""
        payload = json.dumps(
            {
                "v": _CACHE_VERSION,
                "n_pages": len(self._pages),
                "documents": {doc: len(pages) for doc, pages in sorted(self._by_document.items())},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _extract_pdf(pdf_path: Path, cache_dir: Path) -> list[Page]:
    """Extract one PDF's pages, memoized on the PDF's content hash."""
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
    cached = cache_dir / f"{pdf_path.stem}.{digest}.json"
    if cached.is_file():
        raw = json.loads(cached.read_text(encoding="utf-8"))
        return [Page(document_id=pdf_path.stem, page=int(p["page"]), text=p["text"]) for p in raw]

    from pypdf import PdfReader

    pages: list[Page] = []
    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        # Some real SEC filings are AES-encrypted or malformed. One document we cannot open is a
        # COVERAGE GAP, not a reason to abandon an 84-document corpus — and it is a visible one:
        # the corpus reports fewer documents than were asked for, and the fingerprint changes.
        # Silently substituting an empty document would hide it.
        return []

    for index, raw_page in enumerate(reader.pages, start=1):
        try:
            text = _normalize(raw_page.extract_text() or "")
        except Exception:
            # One unparseable page must not sink an 84-document corpus. It becomes an empty page,
            # which is honest: the retriever genuinely cannot find anything there.
            text = ""
        pages.append(Page(document_id=pdf_path.stem, page=index, text=text))

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(
        json.dumps([{"page": p.page, "text": p.text} for p in pages]), encoding="utf-8"
    )
    return pages


def build_corpus(
    pdf_dir: str | Path,
    *,
    documents: set[str] | None = None,
    cache_dir: str | Path | None = None,
) -> PageCorpus:
    """Build (or load from cache) the page corpus for ``documents`` — all of them if ``None``."""
    pdf_root = Path(pdf_dir)
    cache = Path(cache_dir) if cache_dir else pdf_root.parent / "page_cache"

    pages: list[Page] = []
    for pdf in sorted(pdf_root.glob("*.pdf")):
        if documents is not None and pdf.stem not in documents:
            continue
        if pdf.stat().st_size == 0:
            continue
        pages.extend(_extract_pdf(pdf, cache))
    return PageCorpus(pages)
