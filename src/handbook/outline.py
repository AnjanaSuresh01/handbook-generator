"""Planning stage of the plan-then-write pipeline.

No model reliably produces 20,000 coherent words in one call: output degrades
into repetition and drift long before the word count is met. The LongWriter /
AgentWrite result is that you plan first -- an outline where every section
carries its own word budget -- then write each section separately against that
budget. This module produces that plan.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .config import GenerationConfig
from .llm import LLMClient, LLMError

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are a technical editor planning a book-length handbook. "
    "You return only valid JSON, never prose or code fences."
)

_PROMPT = """Plan a handbook titled around this request: {topic}

Base the plan strictly on the source material below. Do not invent topics the
sources cannot support.

SOURCE MATERIAL
---------------
{context}
---------------

Produce a JSON array of {min_sections}-{max_sections} sections that together
form a coherent handbook of about {target_words} words.

Each element must have exactly these keys:
  "title"       - the section heading
  "brief"       - 2-3 sentences saying what this section must cover and how it
                  differs from neighbouring sections
  "key_points"  - array of 3-6 specific points, drawn from the sources
  "words"       - integer word budget for this section

Rules:
- The "words" values must sum to approximately {target_words}.
- No section may exceed 2500 words; split anything larger into two sections.
- Sections must not overlap in content. Each "brief" should make the boundary
  with the previous section explicit.
- Order sections so the handbook reads from foundational to advanced.
- Any conclusion or summary section must come last.

Return a JSON object with exactly two keys:
  "title"    - a specific, descriptive title for the handbook, drawn from what
               the sources actually cover
  "sections" - the array of sections

Return only the JSON object."""

# Titles that belong at the end of a document, whatever the planner decided.
_CLOSING_RE = re.compile(
    r"\b(conclusion|summary|closing|final thoughts|wrap[- ]up|outlook|fazit|zusammenfassung)\b",
    re.IGNORECASE,
)


@dataclass
class Section:
    """One planned section, before it has been written."""

    title: str
    brief: str
    key_points: list[str] = field(default_factory=list)
    words: int = 800
    index: int = 0

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "title": self.title,
            "brief": self.brief,
            "key_points": self.key_points,
            "words": self.words,
        }


@dataclass
class Outline:
    """The full plan for a handbook."""

    topic: str
    sections: list[Section]
    title: str = ""

    @property
    def total_words(self) -> int:
        return sum(section.words for section in self.sections)

    @property
    def display_title(self) -> str:
        """Title for the finished document.

        Falls back to the topic, which is itself a placeholder when the user
        asked for "a handbook" without naming a subject.
        """
        return self.title or self.topic

    def as_dict(self) -> dict:
        return {
            "title": self.display_title,
            "topic": self.topic,
            "total_words": self.total_words,
            "sections": [s.as_dict() for s in self.sections],
        }


def plan_outline(
    client: LLMClient,
    topic: str,
    context: str,
    config: GenerationConfig,
) -> Outline:
    """Ask the model for a sectioned plan, then repair it into a usable one."""
    prompt = _PROMPT.format(
        topic=topic,
        context=context,
        min_sections=config.min_sections,
        max_sections=config.max_sections,
        target_words=config.target_words,
    )
    payload = client.complete_json(prompt, system=_SYSTEM, temperature=0.4)

    title = ""
    if isinstance(payload, dict):
        title = str(payload.get("title") or "").strip()
        # Models still return a bare array often enough to keep handling it.
        payload = payload.get("sections") or payload.get("outline") or []
    if not isinstance(payload, list) or not payload:
        raise LLMError("Planner returned no sections")

    sections = [
        _coerce_section(item, i) for i, item in enumerate(payload) if isinstance(item, dict)
    ]
    if not sections:
        raise LLMError("Planner returned no usable sections")

    sections = rebalance(order_sections(sections), config.target_words)
    return Outline(topic=topic, sections=sections, title=title)


def order_sections(sections: list[Section]) -> list[Section]:
    """Move any conclusion or summary section to the end.

    Planners regularly emit a conclusion in the middle and then keep going,
    which reads as a mistake no matter how good the prose is. Enforcing this
    deterministically is more reliable than asking the model twice.
    """
    body = [s for s in sections if not _CLOSING_RE.search(s.title)]
    closing = [s for s in sections if _CLOSING_RE.search(s.title)]
    # All-closing titles would otherwise empty the handbook.
    ordered = (body + closing) if body else sections
    for i, section in enumerate(ordered):
        section.index = i
    return ordered


def _coerce_section(item: dict, index: int) -> Section:
    key_points = item.get("key_points") or []
    if isinstance(key_points, str):
        key_points = [key_points]

    try:
        words = int(item.get("words") or 800)
    except (TypeError, ValueError):
        words = 800

    return Section(
        title=str(item.get("title") or f"Section {index + 1}").strip(),
        brief=str(item.get("brief") or "").strip(),
        key_points=[str(p).strip() for p in key_points if str(p).strip()],
        words=max(words, 200),
        index=index,
    )


def rebalance(
    sections: list[Section], target_words: int, *, max_section: int = 2500
) -> list[Section]:
    """Scale section budgets so they sum to the target, then cap outliers.

    Planners routinely return budgets summing to half or double the request.
    Scaling proportionally preserves the planner's judgement about relative
    section importance while still honouring the user's word count.
    """
    planned = sum(s.words for s in sections) or 1
    scale = target_words / planned

    for section in sections:
        section.words = max(200, min(int(round(section.words * scale)), max_section))

    # Capping loses words; hand the shortfall to sections with headroom.
    shortfall = target_words - sum(s.words for s in sections)
    if shortfall > 0:
        headroom = [s for s in sections if s.words < max_section]
        if headroom:
            per_section = shortfall // len(headroom)
            for section in headroom:
                section.words = min(section.words + per_section, max_section)

    for i, section in enumerate(sections):
        section.index = i
    return sections
