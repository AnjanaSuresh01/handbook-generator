"""Knowledge store and retrieval.

Two backends behind one interface:

  LightRAG  -- dual-level knowledge-graph retrieval. Preferred, and what the
               assignment asks for: it indexes roughly 60% cheaper than
               Microsoft GraphRAG at about half the query latency.
  Local     -- a dependency-free BM25 index over the same chunks.

The local backend is not a toy fallback: it means the app runs offline with no
API key and no Supabase project, which is what makes the demo reproducible for
anyone who clones the repo.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from .config import StorageConfig
from .ingest import Chunk

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Okapi BM25 defaults; k1 controls term-frequency saturation, b length
# normalisation. These are the standard values and need no tuning here.
_K1 = 1.5
_B = 0.75


def _tokenize(text: str) -> list[str]:
    return [m.group(0).casefold() for m in _TOKEN_RE.finditer(text)]


class KnowledgeStore(Protocol):
    """What the writer needs from a store: index chunks, retrieve context."""

    def index(self, chunks: list[Chunk]) -> None: ...

    def retrieve(self, query: str, max_words: int) -> str: ...


@dataclass
class LocalStore:
    """BM25 retrieval over chunks, entirely in-process."""

    chunks: list[Chunk] = field(default_factory=list)
    _doc_terms: list[Counter] = field(default_factory=list, repr=False)
    _doc_lengths: list[int] = field(default_factory=list, repr=False)
    _df: Counter = field(default_factory=Counter, repr=False)

    def index(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            terms = Counter(_tokenize(chunk.text))
            self.chunks.append(chunk)
            self._doc_terms.append(terms)
            self._doc_lengths.append(sum(terms.values()))
            self._df.update(terms.keys())

    def retrieve(self, query: str, max_words: int) -> str:
        if not self.chunks:
            return ""

        ranked = sorted(
            range(len(self.chunks)), key=lambda i: self._score(query, i), reverse=True
        )

        selected: list[str] = []
        budget = max_words
        for i in ranked:
            words = self.chunks[i].text.split()
            if not words:
                continue
            if len(words) > budget:
                if budget > 50:
                    selected.append(f"[{self.chunks[i].citation}] " + " ".join(words[:budget]))
                break
            selected.append(f"[{self.chunks[i].citation}] {self.chunks[i].text}")
            budget -= len(words)
            if budget <= 0:
                break

        return "\n\n".join(selected)

    def _score(self, query: str, doc_index: int) -> float:
        terms = self._doc_terms[doc_index]
        length = self._doc_lengths[doc_index] or 1
        avg_length = (sum(self._doc_lengths) / len(self._doc_lengths)) or 1
        n_docs = len(self.chunks)

        score = 0.0
        for term in set(_tokenize(query)):
            frequency = terms.get(term, 0)
            if not frequency:
                continue
            df = self._df.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            norm = frequency * (_K1 + 1)
            denom = frequency + _K1 * (1 - _B + _B * length / avg_length)
            score += idf * norm / denom
        return score


@dataclass
class LightRAGStore:
    """Adapter over LightRAG's knowledge-graph retrieval."""

    working_dir: str
    mode: str = "hybrid"
    _rag: object | None = field(default=None, repr=False)

    def _ensure(self) -> object:
        if self._rag is None:
            from lightrag import LightRAG  # type: ignore[import-not-found]

            self._rag = LightRAG(working_dir=self.working_dir)
        return self._rag

    def index(self, chunks: list[Chunk]) -> None:
        rag = self._ensure()
        for chunk in chunks:
            rag.insert(chunk.text)  # type: ignore[attr-defined]

    def retrieve(self, query: str, max_words: int) -> str:
        from lightrag import QueryParam  # type: ignore[import-not-found]

        rag = self._ensure()
        result = rag.query(query, param=QueryParam(mode=self.mode, only_need_context=True))  # type: ignore[attr-defined]
        return " ".join(str(result).split()[:max_words])


def build_store(config: StorageConfig) -> KnowledgeStore:
    """Return the configured store, falling back to local if unavailable."""
    if config.backend == "local":
        return LocalStore()

    try:
        import lightrag  # noqa: F401
    except ImportError:
        log.warning("LightRAG not installed; falling back to the local BM25 store")
        return LocalStore()

    config.working_dir.mkdir(parents=True, exist_ok=True)
    return LightRAGStore(working_dir=str(config.working_dir))
