"""Generate 20,000-word handbooks from uploaded PDFs, through a chat interface."""

from __future__ import annotations

from .assemble import assemble
from .config import Settings, load_settings
from .ingest import Chunk, Document, chunk_document, extract_pdf, ingest_paths
from .llm import LLMClient, LLMError
from .outline import Outline, Section, plan_outline
from .pipeline import Session
from .store import LocalStore, build_store
from .verify import SectionReport, verify_section
from .writer import WriteResult, WrittenSection, write_handbook

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "Document",
    "LLMClient",
    "LLMError",
    "LocalStore",
    "Outline",
    "Section",
    "SectionReport",
    "Session",
    "Settings",
    "WriteResult",
    "WrittenSection",
    "__version__",
    "assemble",
    "build_store",
    "chunk_document",
    "extract_pdf",
    "ingest_paths",
    "load_settings",
    "plan_outline",
    "verify_section",
    "write_handbook",
]
