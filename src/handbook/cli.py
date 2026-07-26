"""Command-line interface.

    handbook generate paper.pdf --topic "Retrieval-Augmented Generation" -o out.md
    handbook plan paper.pdf --topic "RAG"
    handbook ask paper.pdf --question "What problem does this solve?"

Useful without the UI: a 20,000-word run is long enough that you want it in a
terminal you can leave alone, and `plan` lets you check the outline before
spending tokens on the full write.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .llm import LLMError
from .pipeline import Session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="handbook", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def with_pdfs(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("pdfs", nargs="+", type=Path, help="PDF files to index")
        return p

    generate = with_pdfs(sub.add_parser("generate", help="generate a full handbook"))
    generate.add_argument("-t", "--topic", required=True)
    generate.add_argument("-o", "--output", type=Path, default=Path("handbook.md"))
    generate.add_argument("--words", type=int, help="override the target word count")

    plan = with_pdfs(sub.add_parser("plan", help="print the section plan only"))
    plan.add_argument("-t", "--topic", required=True)

    ask = with_pdfs(sub.add_parser("ask", help="ask one question about the PDFs"))
    ask.add_argument("-q", "--question", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    session = Session()
    if args.command == "generate" and args.words:
        object.__setattr__(session.settings.generation, "target_words", args.words)

    try:
        print(session.add_pdfs(args.pdfs), file=sys.stderr)
    except (OSError, RuntimeError) as exc:
        print(f"handbook: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "ask":
            print(session.chat(args.question))
            return 0

        if args.command == "plan":
            print(json.dumps(session.plan(args.topic).as_dict(), indent=2, ensure_ascii=False))
            return 0

        content, result = session.generate_handbook(
            args.topic, on_progress=lambda line: print(line, file=sys.stderr)
        )
        args.output.write_text(content, encoding="utf-8")
        print(json.dumps(result.quality_summary(), indent=2), file=sys.stderr)
        print(f"Wrote {args.output} ({result.total_words:,} words)", file=sys.stderr)
        return 0 if not result.failed else 1

    except (LLMError, ValueError) as exc:
        print(f"handbook: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
