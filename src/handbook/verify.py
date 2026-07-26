"""Quality gate between generation and assembly.

This is what separates a handbook from 20,000 words of slop. Long-form
pipelines fail in three predictable ways, and all three are cheap to detect
without another model call:

  1. Drift      -- the section stops being supported by the source documents.
  2. Repetition -- the section restates an earlier section in new words.
  3. Padding    -- the section pads or truncates instead of meeting its budget.

Sections that fail are regenerated with the failure fed back into the prompt.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .config import VerifyConfig

_TOKEN_RE = re.compile(r"[^\W\d_]{4,}", re.UNICODE)
# The most frequent corpus terms carry no grounding signal; excluding them
# stops "the/and/that" from inflating the score toward a meaningless 1.0.
_STOPWORD_QUANTILE = 60


@dataclass(frozen=True)
class SectionReport:
    """Verification result for a single generated section."""

    index: int
    title: str
    grounding: float
    max_similarity: float
    word_ratio: float
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "title": self.title,
            "grounding": round(self.grounding, 3),
            "max_similarity": round(self.max_similarity, 3),
            "word_ratio": round(self.word_ratio, 3),
            "failures": list(self.failures),
            "passed": self.passed,
        }

    def feedback(self) -> str:
        """Human-readable instruction fed back into the regeneration prompt."""
        notes = {
            "ungrounded": (
                "Too much of the previous attempt was not supported by the source "
                "material. Stay close to the provided context and use its terminology."
            ),
            "repetitive": (
                "The previous attempt repeated content already covered in an earlier "
                "section. Cover only what this section's brief specifies."
            ),
            "too_short": (
                "The previous attempt fell well short of its word budget. Expand with "
                "substantive detail from the sources, not filler."
            ),
            "too_long": "The previous attempt overshot its word budget. Tighten it.",
        }
        return " ".join(notes[f] for f in self.failures if f in notes)


def content_terms(text: str) -> set[str]:
    """Distinct lowercase content terms of four or more letters."""
    return {m.group(0).casefold() for m in _TOKEN_RE.finditer(text)}


def build_corpus_vocabulary(source_text: str) -> tuple[set[str], set[str]]:
    """Return (vocabulary, stopwords) derived from the source documents.

    Stopwords are computed from the corpus itself rather than a fixed list, so
    this works for any language the sources happen to be in.
    """
    counts = Counter(m.group(0).casefold() for m in _TOKEN_RE.finditer(source_text))
    if not counts:
        return set(), set()
    ranked = [term for term, _ in counts.most_common()]
    cutoff = max(1, len(ranked) * _STOPWORD_QUANTILE // 1000)
    return set(ranked), set(ranked[:cutoff])


def grounding_score(section_text: str, vocabulary: set[str], stopwords: set[str]) -> float:
    """Share of the section's distinctive terms that occur in the sources."""
    terms = content_terms(section_text) - stopwords
    if not terms:
        return 1.0
    return len(terms & vocabulary) / len(terms)


def shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    """Overlapping word n-grams, used for near-duplicate detection."""
    words = [m.group(0).casefold() for m in _TOKEN_RE.finditer(text)]
    if len(words) < size:
        return set()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def similarity(a: str, b: str) -> float:
    """Jaccard similarity over 5-word shingles."""
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def verify_section(
    *,
    index: int,
    title: str,
    text: str,
    budget: int,
    vocabulary: set[str],
    stopwords: set[str],
    previous_sections: list[str],
    config: VerifyConfig,
) -> SectionReport:
    """Score one section and list the checks it failed."""
    grounding = grounding_score(text, vocabulary, stopwords)
    max_similarity = max((similarity(text, other) for other in previous_sections), default=0.0)
    word_ratio = len(text.split()) / budget if budget else 1.0

    failures: list[str] = []
    if grounding < config.min_grounding:
        failures.append("ungrounded")
    if max_similarity > config.max_similarity:
        failures.append("repetitive")
    if word_ratio < config.min_word_ratio:
        failures.append("too_short")
    elif word_ratio > 2.0:
        failures.append("too_long")

    return SectionReport(
        index=index,
        title=title,
        grounding=grounding,
        max_similarity=max_similarity,
        word_ratio=word_ratio,
        failures=tuple(failures),
    )
