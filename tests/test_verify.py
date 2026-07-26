from handbook.config import VerifyConfig
from handbook.verify import (
    build_corpus_vocabulary,
    grounding_score,
    similarity,
    verify_section,
)

SOURCE = """
Retrieval-augmented generation combines a retriever with a generator. The
retriever selects passages from a corpus using dense embeddings, and the
generator conditions on those passages when producing an answer. This reduces
hallucination because the model has grounded evidence available at inference
time. Chunking strategy and reranking both materially affect retrieval quality.
""" * 3

CONFIG = VerifyConfig()


def vocab():
    return build_corpus_vocabulary(SOURCE)


class TestGrounding:
    def test_text_from_the_source_scores_high(self):
        vocabulary, stopwords = vocab()
        text = "The retriever selects passages using dense embeddings before generation."
        assert grounding_score(text, vocabulary, stopwords) > 0.7

    def test_unrelated_text_scores_low(self):
        vocabulary, stopwords = vocab()
        text = "Volcanic basalt formations dominate Icelandic coastal geomorphology entirely."
        assert grounding_score(text, vocabulary, stopwords) < 0.3

    def test_empty_text_does_not_crash(self):
        vocabulary, stopwords = vocab()
        assert grounding_score("", vocabulary, stopwords) == 1.0


class TestSimilarity:
    def test_identical_text_is_one(self):
        text = "the retriever selects passages from a corpus using dense embeddings"
        assert similarity(text, text) == 1.0

    def test_unrelated_text_is_zero(self):
        assert similarity("alpha beta gamma delta epsilon", "one two three four five") == 0.0

    def test_short_text_does_not_crash(self):
        assert similarity("too short", "also short") == 0.0


class TestVerifySection:
    def _verify(self, text, budget=50, previous=()):
        vocabulary, stopwords = vocab()
        return verify_section(
            index=0,
            title="Retrieval",
            text=text,
            budget=budget,
            vocabulary=vocabulary,
            stopwords=stopwords,
            previous_sections=list(previous),
            config=CONFIG,
        )

    def test_good_section_passes(self):
        text = " ".join(
            ["The retriever selects passages from the corpus using dense embeddings, "
             "and the generator conditions on those passages to reduce hallucination."] * 4
        )
        report = self._verify(text, budget=len(text.split()))
        assert report.passed
        assert report.failures == ()

    def test_ungrounded_section_is_flagged(self):
        text = " ".join(
            ["Volcanic basalt formations dominate Icelandic coastal geomorphology."] * 8
        )
        report = self._verify(text, budget=len(text.split()))
        assert "ungrounded" in report.failures

    def test_repetitive_section_is_flagged(self):
        text = " ".join(
            ["The retriever selects passages from the corpus using dense embeddings."] * 8
        )
        report = self._verify(text, budget=len(text.split()), previous=[text])
        assert "repetitive" in report.failures

    def test_short_section_is_flagged(self):
        report = self._verify("The retriever selects passages.", budget=500)
        assert "too_short" in report.failures

    def test_overlong_section_is_flagged(self):
        text = " ".join(["retriever passages embeddings generator corpus"] * 100)
        report = self._verify(text, budget=50)
        assert "too_long" in report.failures

    def test_feedback_is_actionable(self):
        report = self._verify("Short.", budget=500)
        assert "word budget" in report.feedback()

    def test_report_serialises(self):
        payload = self._verify("The retriever selects passages.", budget=500).as_dict()
        assert payload["passed"] is False
        assert "grounding" in payload
