"""Writing stage: generate each planned section, verify it, retry on failure.

Sections are written one at a time against their own word budget, each seeing
the tail of the previous section for continuity but not the whole handbook --
that is what keeps total output far beyond any single-call limit while staying
inside the context window.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from .config import GenerationConfig, VerifyConfig
from .llm import LLMClient
from .outline import Outline, Section
from .verify import SectionReport, build_corpus_vocabulary, verify_section

log = logging.getLogger(__name__)

Retriever = Callable[[str, int], str]
ProgressCallback = Callable[[str], None]

_SYSTEM = (
    "You are writing one section of a technical handbook. "
    "Write substantive, specific prose grounded in the supplied source material. "
    "Never pad, never restate the section brief back at the reader, and never "
    "announce what you are about to do."
)

_PROMPT = """HANDBOOK TOPIC: {topic}

SECTION {number} OF {total}: {title}

WHAT THIS SECTION MUST COVER
{brief}

KEY POINTS TO ADDRESS
{key_points}

WORD BUDGET: approximately {words} words.

SOURCE MATERIAL
---------------
{context}
---------------
{continuity}{feedback}
Write the body of this section in Markdown. Start with "## {title}" and use
"###" for any subheadings. Do not write a preamble, a summary of the whole
handbook, or a conclusion referring to other sections. Write only this section."""

_CONTINUITY = """
The previous section ended with:
"...{tail}"
Continue naturally from there without repeating it.
"""

_FEEDBACK = """
IMPORTANT -- your previous attempt at this section was rejected.
{feedback}
"""


@dataclass
class WrittenSection:
    """A generated section together with its verification result."""

    section: Section
    text: str
    report: SectionReport
    attempts: int = 1

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class WriteResult:
    """Everything produced by a full handbook run."""

    outline: Outline
    sections: list[WrittenSection] = field(default_factory=list)

    @property
    def total_words(self) -> int:
        return sum(s.word_count for s in self.sections)

    @property
    def regenerated(self) -> int:
        return sum(1 for s in self.sections if s.attempts > 1)

    @property
    def failed(self) -> list[WrittenSection]:
        return [s for s in self.sections if not s.report.passed]

    def quality_summary(self) -> dict:
        count = len(self.sections) or 1
        return {
            "sections": len(self.sections),
            "total_words": self.total_words,
            "regenerated_sections": self.regenerated,
            "sections_failing_after_retries": len(self.failed),
            "mean_grounding": round(
                sum(s.report.grounding for s in self.sections) / count, 3
            ),
            "max_cross_section_similarity": round(
                max((s.report.max_similarity for s in self.sections), default=0.0), 3
            ),
        }


def write_handbook(
    client: LLMClient,
    outline: Outline,
    *,
    retrieve: Retriever,
    source_text: str,
    generation: GenerationConfig,
    verification: VerifyConfig,
    on_progress: ProgressCallback | None = None,
) -> WriteResult:
    """Generate every section of ``outline``, verifying and retrying as needed."""
    vocabulary, stopwords = build_corpus_vocabulary(source_text)
    result = WriteResult(outline=outline)
    written_texts: list[str] = []

    total = len(outline.sections)
    for section in outline.sections:
        if on_progress:
            on_progress(f"Writing section {section.index + 1}/{total}: {section.title}")

        text, report, attempts = _write_one(
            client,
            outline=outline,
            section=section,
            total=total,
            retrieve=retrieve,
            vocabulary=vocabulary,
            stopwords=stopwords,
            previous_texts=written_texts,
            tail=_tail(written_texts[-1]) if written_texts else "",
            verification=verification,
        )

        written_texts.append(text)
        result.sections.append(WrittenSection(section, text, report, attempts))

        if on_progress:
            status = "ok" if report.passed else f"kept with issues: {', '.join(report.failures)}"
            on_progress(
                f"  section {section.index + 1} done -- {len(text.split())} words, "
                f"grounding {report.grounding:.2f}, {status}"
            )

    return result


def _write_one(
    client: LLMClient,
    *,
    outline: Outline,
    section: Section,
    total: int,
    retrieve: Retriever,
    vocabulary: set[str],
    stopwords: set[str],
    previous_texts: list[str],
    tail: str,
    verification: VerifyConfig,
) -> tuple[str, SectionReport, int]:
    """Generate one section, retrying while it fails verification."""
    query = f"{section.title}. {section.brief} {' '.join(section.key_points)}"
    context = retrieve(query, section.words * 2)

    best: tuple[str, SectionReport] | None = None
    feedback = ""

    for attempt in range(1, verification.max_retries + 2):
        prompt = _PROMPT.format(
            topic=outline.display_title,
            number=section.index + 1,
            total=total,
            title=section.title,
            brief=section.brief,
            key_points="\n".join(f"- {p}" for p in section.key_points) or "- (none specified)",
            words=section.words,
            context=context,
            continuity=_CONTINUITY.format(tail=tail) if tail else "",
            feedback=_FEEDBACK.format(feedback=feedback) if feedback else "",
        )
        text = client.complete(prompt, system=_SYSTEM, temperature=0.7).strip()

        report = verify_section(
            index=section.index,
            title=section.title,
            text=text,
            budget=section.words,
            vocabulary=vocabulary,
            stopwords=stopwords,
            previous_sections=previous_texts,
            config=verification,
        )

        if report.passed:
            return text, report, attempt

        # Keep the least-bad attempt so a stubborn section still yields output.
        if best is None or len(report.failures) < len(best[1].failures):
            best = (text, report)

        feedback = report.feedback()
        log.info("Section %d failed (%s), retrying", section.index + 1, ", ".join(report.failures))

    assert best is not None
    return best[0], best[1], verification.max_retries + 1


def _tail(text: str, words: int = 40) -> str:
    return " ".join(text.split()[-words:])
