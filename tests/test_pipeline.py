"""End-to-end tests with a stub LLM, so the suite runs offline and free."""

from __future__ import annotations

import json

import pytest

from handbook.assemble import assemble, slugify
from handbook.config import GenerationConfig, Settings, VerifyConfig
from handbook.ingest import Chunk, Document, chunk_document, clean_text
from handbook.llm import LLMError, parse_json_block
from handbook.outline import plan_outline, rebalance
from handbook.pipeline import Session
from handbook.store import LocalStore
from handbook.writer import write_handbook

SOURCE = (
    "Retrieval-augmented generation combines a retriever with a generator. "
    "The retriever selects passages from a corpus using dense embeddings. "
    "The generator conditions on those passages when producing an answer. "
    "Chunking strategy and reranking both affect retrieval quality materially. "
) * 20


class StubLLM:
    """Returns a valid plan, then grounded section text of the requested length."""

    def __init__(self, sections: int = 3):
        self.sections = sections
        self.calls = 0

    def complete(self, prompt, *, system=None, temperature=0.7, max_tokens=None):
        self.calls += 1
        words = 400
        for line in prompt.splitlines():
            if line.startswith("WORD BUDGET"):
                words = int("".join(c for c in line if c.isdigit()) or 400)
        sentence = (
            "The retriever selects passages from the corpus using dense embeddings "
            "and the generator conditions on them to reduce hallucination. "
        )
        body = (sentence * (words // len(sentence.split()) + 2)).strip()
        return f"## Section\n\n{body}"

    def complete_json(self, prompt, *, system=None, **kwargs):
        self.calls += 1
        return [
            {
                "title": f"Section {i + 1}",
                "brief": f"Covers aspect {i + 1} of retrieval-augmented generation.",
                "key_points": ["retriever", "embeddings", "grounding"],
                "words": 400,
            }
            for i in range(self.sections)
        ]


class TestIngest:
    def test_clean_text_rejoins_hyphenated_linebreaks(self):
        assert "Retrieval-Augmented" not in clean_text("Retrieval-\nAugmented")
        assert clean_text("Retrieval-\nAugmented") == "RetrievalAugmented"

    def test_chunking_overlaps(self):
        doc = Document("a.pdf", " ".join(str(i) for i in range(1000)), 1, "test")
        chunks = chunk_document(doc, chunk_words=100, overlap_words=20)
        assert len(chunks) > 1
        assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]

    def test_empty_document_yields_no_chunks(self):
        assert chunk_document(Document("a.pdf", "", 0, "test")) == []

    def test_overlap_must_be_smaller_than_chunk(self):
        doc = Document("a.pdf", "one two three", 1, "test")
        with pytest.raises(ValueError):
            chunk_document(doc, chunk_words=10, overlap_words=10)

    def test_citation_format(self):
        assert Chunk("paper.pdf", 3, "x").citation == "paper.pdf#chunk3"


class TestLocalStore:
    def _store(self):
        store = LocalStore()
        store.index(
        chunk_document(
            Document("rag.pdf", SOURCE, 1, "test"), chunk_words=60, overlap_words=15
        )
    )
        return store

    def test_retrieval_returns_relevant_text(self):
        context = self._store().retrieve("dense embeddings retriever", 200)
        assert "embeddings" in context.lower()
        assert "rag.pdf#chunk" in context

    def test_retrieval_respects_word_budget(self):
        context = self._store().retrieve("retriever", 80)
        assert len(context.split()) <= 100

    def test_empty_store_returns_empty_string(self):
        assert LocalStore().retrieve("anything", 100) == ""


class TestJSONParsing:
    def test_plain_json(self):
        assert parse_json_block('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert parse_json_block('```json\n[{"a": 1}]\n```') == [{"a": 1}]

    def test_json_wrapped_in_prose(self):
        assert parse_json_block('Sure! Here you go:\n[{"a": 1}]\nHope that helps.') == [{"a": 1}]

    def test_unparseable_raises(self):
        with pytest.raises(LLMError):
            parse_json_block("no json at all here")


class TestOutline:
    def test_budgets_are_rescaled_to_target(self):
        outline = plan_outline(
            StubLLM(sections=10), "RAG", SOURCE, GenerationConfig(target_words=20_000)
        )
        assert 18_000 <= outline.total_words <= 22_000

    def test_no_section_exceeds_the_cap(self):
        outline = plan_outline(
            StubLLM(sections=4), "RAG", SOURCE, GenerationConfig(target_words=20_000)
        )
        assert all(s.words <= 2500 for s in outline.sections)

    def test_rebalance_scales_up_and_down(self):
        from handbook.outline import Section

        sections = [Section(title=f"S{i}", brief="", words=100) for i in range(10)]
        rebalance(sections, 5000)
        assert sum(s.words for s in sections) == pytest.approx(5000, rel=0.1)

    def test_empty_plan_raises(self):
        class Empty(StubLLM):
            def complete_json(self, prompt, *, system=None, **kwargs):
                return []

        with pytest.raises(LLMError):
            plan_outline(Empty(), "RAG", SOURCE, GenerationConfig())


class TestWriteAndAssemble:
    def _run(self):
        llm = StubLLM(sections=3)
        store = LocalStore()
        store.index(
        chunk_document(
            Document("rag.pdf", SOURCE, 1, "test"), chunk_words=60, overlap_words=15
        )
    )
        outline = plan_outline(llm, "RAG", SOURCE, GenerationConfig(target_words=1200))
        return write_handbook(
            llm,
            outline,
            retrieve=store.retrieve,
            source_text=SOURCE,
            generation=GenerationConfig(target_words=1200),
            verification=VerifyConfig(),
        )

    def test_every_section_gets_written(self):
        result = self._run()
        assert len(result.sections) == 3
        assert result.total_words > 0

    def test_quality_summary_is_complete(self):
        summary = self._run().quality_summary()
        assert set(summary) == {
            "sections",
            "total_words",
            "regenerated_sections",
            "sections_failing_after_retries",
            "mean_grounding",
            "max_cross_section_similarity",
        }

    def test_assembled_document_has_structure(self):
        result = self._run()
        document = assemble(result, [Document("rag.pdf", SOURCE, 10, "test")])
        assert document.startswith("# RAG")
        assert "## Table of Contents" in document
        assert "## Sources" in document
        assert "## Generation Quality Report" in document
        assert "rag.pdf" in document

    def test_progress_callback_fires(self):
        lines: list[str] = []
        llm = StubLLM(sections=2)
        store = LocalStore()
        store.index(
        chunk_document(
            Document("rag.pdf", SOURCE, 1, "test"), chunk_words=60, overlap_words=15
        )
    )
        outline = plan_outline(llm, "RAG", SOURCE, GenerationConfig(target_words=800))
        write_handbook(
            llm,
            outline,
            retrieve=store.retrieve,
            source_text=SOURCE,
            generation=GenerationConfig(target_words=800),
            verification=VerifyConfig(),
            on_progress=lines.append,
        )
        assert any("Writing section" in line for line in lines)

    def test_slugify_makes_valid_anchors(self):
        assert slugify("Retrieval-Augmented Generation!") == "retrieval-augmented-generation"


class TestSession:
    def _session(self):
        session = Session(settings=Settings())
        session.client = StubLLM(sections=3)
        session.documents.append(Document("rag.pdf", SOURCE, 1, "test"))
        chunks = chunk_document(session.documents[0], chunk_words=60, overlap_words=15)
        session.chunks.extend(chunks)
        session.store.index(chunks)
        return session

    def test_handbook_requires_documents(self):
        session = Session(settings=Settings())
        session.client = StubLLM()
        with pytest.raises(ValueError, match="Upload at least one PDF"):
            session.generate_handbook("RAG")

    def test_chat_without_documents_is_graceful(self):
        session = Session(settings=Settings())
        session.client = StubLLM()
        assert "Upload a PDF first" in session.chat("what is rag?")

    def test_generate_returns_markdown_and_result(self):
        content, result = self._session().generate_handbook("RAG")
        assert content.startswith("# RAG")
        assert result.total_words > 0

    def test_chat_records_history(self):
        session = self._session()
        session.chat("What is a retriever?")
        assert len(session.history) == 1


class TestUI:
    def test_ui_builds(self):
        """Construct the real Gradio layout.

        Importing app.py is not enough: components are only instantiated inside
        build_ui(), so an incompatible Gradio release fails here rather than on
        the user's screen at launch.
        """
        import gradio as gr

        from app import build_ui

        assert isinstance(build_ui(), gr.Blocks)


class TestHandbookIntentDetection:
    @pytest.mark.parametrize(
        "message,expected",
        [
            ("generate a handbook on RAG", "RAG"),
            ("Create a handbook about vector databases", "vector databases"),
            ("write me a handbook covering chunking", "chunking"),
            ("please build a handbook: embeddings", "embeddings"),
            ("generate a handbook", "the uploaded documents"),
        ],
    )
    def test_detects_requests(self, message, expected):
        from app import detect_handbook_request

        assert detect_handbook_request(message) == expected

    @pytest.mark.parametrize(
        "message",
        ["what is a handbook?", "summarise the paper", "who wrote this?"],
    )
    def test_ignores_ordinary_questions(self, message):
        from app import detect_handbook_request

        assert detect_handbook_request(message) is None


def test_quality_report_is_json_serialisable():
    llm = StubLLM(sections=2)
    store = LocalStore()
    store.index(
        chunk_document(
            Document("rag.pdf", SOURCE, 1, "test"), chunk_words=60, overlap_words=15
        )
    )
    outline = plan_outline(llm, "RAG", SOURCE, GenerationConfig(target_words=800))
    result = write_handbook(
        llm,
        outline,
        retrieve=store.retrieve,
        source_text=SOURCE,
        generation=GenerationConfig(target_words=800),
        verification=VerifyConfig(),
    )
    json.dumps(result.quality_summary())
    json.dumps([s.report.as_dict() for s in result.sections])
