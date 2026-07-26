"""Gradio chat interface.

    python app.py

Everything happens through conversation: upload PDFs, ask questions, then ask
for a handbook. Handbook generation streams progress into the chat because a
20,000-word run takes minutes and a silent spinner is indistinguishable from a
hang.
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

import gradio as gr

from handbook.assemble import export_markdown
from handbook.llm import LLMError
from handbook.pipeline import Session

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# "write a handbook on X" / "generate handbook about X" / "create a handbook: X"
_HANDBOOK_RE = re.compile(
    r"\b(?:generate|create|write|make|build|produce)\b.{0,30}?\bhandbook\b"
    r"(?:\s*(?:on|about|for|covering|:)\s*(?P<topic>.+))?",
    re.IGNORECASE | re.DOTALL,
)

WELCOME = """### Handbook Generator

1. Upload one or more PDFs
2. Ask questions about them
3. Say **"generate a handbook on <topic>"** for a 20,000+ word document

Every section is checked for source grounding, repetition against earlier
sections, and word budget before it reaches the final document. Sections that
fail are rewritten.
"""


def detect_handbook_request(message: str) -> str | None:
    """Return the requested topic, or None if this is an ordinary question."""
    match = _HANDBOOK_RE.search(message)
    if not match:
        return None
    topic = (match.group("topic") or "").strip().strip("\"'")
    return topic or "the uploaded documents"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Handbook Generator", fill_height=True) as demo:
        session = gr.State(None)
        handbook_path = gr.State(None)

        gr.Markdown(WELCOME)

        with gr.Row():
            with gr.Column(scale=1):
                uploads = gr.File(
                    label="PDF documents",
                    file_types=[".pdf"],
                    file_count="multiple",
                    height=180,
                )
                status = gr.Markdown("*No documents indexed yet.*")
                download = gr.File(label="Generated handbook", visible=False, interactive=False)

            with gr.Column(scale=2):
                # Gradio 6 dropped the `type` argument; {"role", "content"}
                # message dicts are the only supported format now.
                chatbot = gr.Chatbot(label="Chat", height=520)
                message = gr.Textbox(
                    placeholder=(
                        "Ask a question, or: generate a handbook on "
                        "retrieval-augmented generation"
                    ),
                    show_label=False,
                    autofocus=True,
                )
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear chat")

        # -- handlers ------------------------------------------------------

        def on_upload(files, state):
            if not files:
                return state, "*No documents indexed yet.*"
            state = state or Session()
            paths = [f.name if hasattr(f, "name") else str(f) for f in files]
            try:
                summary = state.add_pdfs(paths)
            except Exception as exc:  # surfaced in the UI, not the console
                return state, f"**Upload failed:** {exc}"
            return state, summary.replace("\n", "\n\n")

        def on_send(user_message, history, state, path):
            user_message = (user_message or "").strip()
            history = history or []
            if not user_message:
                yield history, state, path, gr.update(), ""
                return

            state = state or Session()
            history = history + [{"role": "user", "content": user_message}]
            yield history, state, path, gr.update(), ""

            topic = detect_handbook_request(user_message)
            try:
                if topic is None:
                    reply = state.chat(user_message)
                    history = history + [{"role": "assistant", "content": reply}]
                    yield history, state, path, gr.update(), ""
                    return

                # Handbook run: stream progress lines as they arrive.
                progress_lines: list[str] = []
                history = history + [{"role": "assistant", "content": "Starting..."}]

                def on_progress(line: str) -> None:
                    progress_lines.append(line)

                content, result = state.generate_handbook(topic, on_progress=on_progress)

                out_dir = Path(tempfile.gettempdir()) / "handbooks"
                out_dir.mkdir(exist_ok=True)
                safe = re.sub(r"[^\w-]+", "-", topic)[:60].strip("-") or "handbook"
                path = export_markdown(content, str(out_dir / f"{safe}.md"))

                summary = result.quality_summary()
                history[-1] = {
                    "role": "assistant",
                    "content": (
                        "\n".join(f"- {line}" for line in progress_lines)
                        + f"\n\n**Handbook ready — {summary['total_words']:,} words across "
                        f"{summary['sections']} sections.**\n\n"
                        f"- Mean grounding in your documents: {summary['mean_grounding']:.2f}\n"
                        f"- Highest cross-section similarity: "
                        f"{summary['max_cross_section_similarity']:.2f}\n"
                        f"- Sections rewritten after failing verification: "
                        f"{summary['regenerated_sections']}\n\n"
                        "Download it from the panel on the left."
                    ),
                }
                yield history, state, path, gr.update(value=path, visible=True), ""

            except (LLMError, ValueError) as exc:
                history = history + [{"role": "assistant", "content": f"**Error:** {exc}"}]
                yield history, state, path, gr.update(), ""

        uploads.change(on_upload, [uploads, session], [session, status])
        send.click(
            on_send,
            [message, chatbot, session, handbook_path],
            [chatbot, session, handbook_path, download, message],
        )
        message.submit(
            on_send,
            [message, chatbot, session, handbook_path],
            [chatbot, session, handbook_path, download, message],
        )
        clear.click(lambda: [], outputs=chatbot)

    return demo


if __name__ == "__main__":
    build_ui().launch()
