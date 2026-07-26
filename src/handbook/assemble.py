"""Assembly stage: stitch verified sections into a finished document.

Adds the structure a handbook needs and the generator cannot produce in a
single pass -- title, table of contents, source citations and a quality report
recording how the document was verified.
"""

from __future__ import annotations

import re
from datetime import date

from .ingest import Document
from .writer import WriteResult

_HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def slugify(text: str) -> str:
    """GitHub-style anchor for a heading."""
    slug = re.sub(r"[^\w\s-]", "", text.casefold())
    return re.sub(r"[\s_]+", "-", slug).strip("-")


def build_toc(result: WriteResult) -> str:
    lines = ["## Table of Contents", ""]
    for written in result.sections:
        title = written.section.title
        lines.append(f"{written.section.index + 1}. [{title}](#{slugify(title)})")
    return "\n".join(lines)


def build_sources(documents: list[Document]) -> str:
    if not documents:
        return ""
    lines = ["## Sources", "", "This handbook was generated from the following documents:", ""]
    for doc in documents:
        lines.append(f"- **{doc.name}** — {doc.pages} pages, {doc.word_count:,} words")
    return "\n".join(lines)


def build_quality_report(result: WriteResult) -> str:
    """Per-section verification table.

    Included in the output on purpose: a generated document that reports how it
    was checked is worth more than one that simply asserts it is correct.
    """
    summary = result.quality_summary()
    lines = [
        "## Generation Quality Report",
        "",
        f"- Sections: {summary['sections']}",
        f"- Total words: {summary['total_words']:,}",
        f"- Sections regenerated after failing verification: {summary['regenerated_sections']}",
        f"- Sections still failing after retries: {summary['sections_failing_after_retries']}",
        f"- Mean grounding in source material: {summary['mean_grounding']:.2f}",
        f"- Highest cross-section similarity: {summary['max_cross_section_similarity']:.2f}",
        "",
        "| # | Section | Words | Grounding | Max similarity | Status |",
        "|---|---------|-------|-----------|----------------|--------|",
    ]
    for written in result.sections:
        report = written.report
        status = "pass" if report.passed else ", ".join(report.failures)
        lines.append(
            f"| {report.index + 1} | {report.title} | {written.word_count} | "
            f"{report.grounding:.2f} | {report.max_similarity:.2f} | {status} |"
        )
    lines += [
        "",
        "**Grounding** is the share of a section's distinctive terms that also occur in "
        "the uploaded documents. **Max similarity** is the highest 5-gram Jaccard overlap "
        "with any earlier section, so a low value means sections are not restating each "
        "other. Both are computed without extra model calls.",
    ]
    return "\n".join(lines)


def assemble(
    result: WriteResult,
    documents: list[Document],
    *,
    include_quality_report: bool = True,
) -> str:
    """Produce the final Markdown handbook."""
    title = result.outline.display_title.strip().rstrip(".")
    parts = [
        f"# {title}",
        "",
        f"*Generated on {date.today().isoformat()} from "
        f"{len(documents)} source document{'s' if len(documents) != 1 else ''}. "
        f"{result.total_words:,} words.*",
        "",
        build_toc(result),
        "",
        "---",
        "",
    ]

    for written in result.sections:
        parts.append(_ensure_heading(written.text, written.section.title))
        parts.append("")

    parts += ["---", "", build_sources(documents), ""]
    if include_quality_report:
        parts += [build_quality_report(result), ""]

    return "\n".join(parts).strip() + "\n"


def _ensure_heading(text: str, title: str) -> str:
    """Guarantee the section starts with its own H2, whatever the model did."""
    text = text.strip()
    if _HEADING_RE.search(text[:200]):
        return text
    return f"## {title}\n\n{text}"


def export_markdown(content: str, path: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path
