"""PDF ingestion: extract text, then chunk it for indexing.

Parser selection is deliberate. `liteparse` is preferred because it detects
whether a page actually needs OCR and preserves tables as Markdown, which
matters when the handbook has to cite figures from the source. `pypdf` is the
always-available fallback so the app runs with no extra system dependencies.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")
# Hyphen splitting a word across a line break, e.g. "Retrieval-\nAugmented".
_DEHYPHEN_RE = re.compile(r"(\w)-\n(\w)")


@dataclass(frozen=True)
class Document:
    """One ingested PDF."""

    name: str
    text: str
    pages: int
    parser: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class Chunk:
    """A slice of a document, sized for embedding and retrieval."""

    doc_name: str
    index: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc_name}#chunk{self.index}"


def extract_pdf(path: str | Path) -> Document:
    """Extract text from a PDF, preferring the richer parser when installed."""
    path = Path(path)
    for extractor in (_extract_liteparse, _extract_pypdf):
        try:
            return extractor(path)
        except ImportError:
            continue
    raise RuntimeError("No PDF parser available. Install pypdf or liteparse.")


def _extract_liteparse(path: Path) -> Document:
    from liteparse import parse  # type: ignore[import-not-found]

    result = parse(str(path))
    text = getattr(result, "markdown", None) or str(result)
    pages = len(getattr(result, "pages", []) or [])
    return Document(path.name, clean_text(text), pages, "liteparse")


def _extract_pypdf(path: Path) -> Document:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception as exc:  # a single broken page must not kill the upload
            log.warning("Skipping unreadable page in %s: %s", path.name, exc)
    return Document(path.name, clean_text("\n\n".join(parts)), len(reader.pages), "pypdf")


def clean_text(text: str) -> str:
    """Normalise extractor output: join hyphenated line breaks, collapse space."""
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = _DEHYPHEN_RE.sub(r"\1\2", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return _BLANKLINES_RE.sub("\n\n", text).strip()


def chunk_document(
    document: Document,
    *,
    chunk_words: int = 400,
    overlap_words: int = 60,
) -> list[Chunk]:
    """Split a document into overlapping word windows.

    Overlap exists so a fact sitting on a chunk boundary is still retrievable
    from at least one complete chunk.
    """
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    words = document.text.split()
    if not words:
        return []

    step = chunk_words - overlap_words
    chunks: list[Chunk] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        if not window:
            break
        chunks.append(Chunk(document.name, len(chunks), " ".join(window)))
        if start + chunk_words >= len(words):
            break
    return chunks


def ingest_paths(paths: list[str | Path], **chunk_kwargs) -> tuple[list[Document], list[Chunk]]:
    """Extract and chunk several PDFs in one call."""
    documents = [extract_pdf(path) for path in paths]
    chunks = [c for doc in documents for c in chunk_document(doc, **chunk_kwargs)]
    return documents, chunks
