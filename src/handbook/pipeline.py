"""Orchestration: the object the UI talks to.

Holds one session's state -- uploaded documents, the knowledge store, the chat
history -- and exposes the three things the assignment asks for: ingest PDFs,
chat about them, generate a handbook.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .assemble import assemble
from .config import Settings, load_settings
from .ingest import Chunk, Document, ingest_paths
from .llm import LLMClient
from .outline import Outline, plan_outline
from .store import KnowledgeStore, build_store
from .writer import WriteResult, write_handbook

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]

_CHAT_SYSTEM = (
    "You answer questions using only the supplied source material. "
    "Cite the bracketed chunk references you used, like [paper.pdf#chunk3]. "
    "If the sources do not contain the answer, say so plainly instead of guessing."
)

_CHAT_PROMPT = """SOURCE MATERIAL
---------------
{context}
---------------

{history}QUESTION: {question}

Answer using only the source material above, citing the chunk references you relied on."""


@dataclass
class Session:
    """One user's working state."""

    settings: Settings = field(default_factory=load_settings)
    documents: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    history: list[tuple[str, str]] = field(default_factory=list)
    store: KnowledgeStore | None = field(default=None, repr=False)
    client: LLMClient | None = field(default=None, repr=False)
    last_result: WriteResult | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.store is None:
            self.store = build_store(self.settings.storage)
        if self.client is None:
            self.client = LLMClient(self.settings.llm)

    # -- ingest ------------------------------------------------------------

    @property
    def indexed(self) -> bool:
        return bool(self.chunks)

    @property
    def source_text(self) -> str:
        return "\n\n".join(doc.text for doc in self.documents)

    def add_pdfs(self, paths: list[str | Path]) -> str:
        """Extract, chunk and index PDFs. Returns a human-readable summary."""
        documents, chunks = ingest_paths(paths)
        if not chunks:
            return "No extractable text found in those files."

        assert self.store is not None
        self.store.index(chunks)
        self.documents.extend(documents)
        self.chunks.extend(chunks)

        lines = [f"Indexed {len(documents)} document(s), {len(chunks)} chunks:"]
        lines += [
            f"- {doc.name}: {doc.pages} pages, {doc.word_count:,} words (via {doc.parser})"
            for doc in documents
        ]
        lines.append("\nAsk me a question, or say: generate a handbook on <topic>")
        return "\n".join(lines)

    # -- chat --------------------------------------------------------------

    def chat(self, question: str, *, context_words: int = 2000) -> str:
        """Answer a question against the indexed sources."""
        if not self.indexed:
            return "Upload a PDF first — I have nothing indexed yet."

        assert self.store is not None and self.client is not None
        context = self.store.retrieve(question, context_words)
        answer = self.client.complete(
            _CHAT_PROMPT.format(
                context=context,
                history=self._recent_history(),
                question=question,
            ),
            system=_CHAT_SYSTEM,
            temperature=0.3,
        )
        self.history.append((question, answer))
        return answer

    def _recent_history(self, turns: int = 3) -> str:
        if not self.history:
            return ""
        recent = self.history[-turns:]
        lines = ["EARLIER IN THIS CONVERSATION"]
        lines += [f"Q: {q}\nA: {a[:400]}" for q, a in recent]
        return "\n".join(lines) + "\n\n"

    # -- handbook ----------------------------------------------------------

    def plan(self, topic: str, *, context_words: int = 4000) -> Outline:
        """Produce the section plan without writing anything yet."""
        assert self.store is not None and self.client is not None
        context = self.store.retrieve(topic, context_words)
        return plan_outline(self.client, topic, context, self.settings.generation)

    def generate_handbook(
        self,
        topic: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[str, WriteResult]:
        """Plan, write, verify and assemble a full handbook."""
        if not self.indexed:
            raise ValueError("Upload at least one PDF before generating a handbook.")

        assert self.store is not None and self.client is not None

        if on_progress:
            on_progress("Planning outline...")
        outline = self.plan(topic)
        if on_progress:
            on_progress(
                f"Planned {len(outline.sections)} sections, "
                f"{outline.total_words:,} words budgeted."
            )

        result = write_handbook(
            self.client,
            outline,
            retrieve=self.store.retrieve,
            source_text=self.source_text,
            generation=self.settings.generation,
            verification=self.settings.verify,
            on_progress=on_progress,
        )
        self.last_result = result

        if on_progress:
            usage = self.client.usage
            on_progress(
                f"Done: {result.total_words:,} words, "
                f"{result.regenerated} section(s) regenerated, "
                f"{usage.calls} model calls, {usage.total_tokens:,} tokens."
            )

        return assemble(result, self.documents), result
