# Handbook Generator

Upload PDFs, chat about them, and generate a **20,000+ word structured handbook**
grounded in those documents — all through a chat interface.

![CI](https://github.com/AnjanaSuresh01/handbook-generator/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## The actual problem

No language model writes 20,000 coherent words in one call. Ask for it and
output degrades into repetition and drift thousands of words before the target.

So this follows the **plan-then-write** approach from
[LongWriter](https://github.com/THUDM/LongWriter) (ICLR 2025): plan an outline
where every section carries its own word budget, then write each section
separately against that budget.

That gets you to 20,000 words. It does **not** get you to 20,000 *good* words —
sections still drift off-source, silently restate each other, and pad to hit
their count.

**So there is a verification pass between generation and assembly.** Every
section is scored on three axes, and sections that fail are regenerated with the
specific failure fed back into the prompt:

| Check | Detects | How |
|---|---|---|
| **Grounding** | Section drifting away from your PDFs | Share of the section's distinctive terms that occur in the sources |
| **Similarity** | Section restating an earlier one | Highest 5-gram Jaccard overlap against every prior section |
| **Word ratio** | Padding or truncation | Actual words vs. the planned budget |

None of these need an extra model call, so verification is effectively free.
The finished handbook ships with a **quality report** showing the numbers per
section — a generated document that reports how it was checked is worth more
than one that just asserts it is correct.

Stopwords for the grounding score are computed **from your uploaded documents**,
not from a fixed English list, so this works whatever language your sources are
in.

## Quickstart

```bash
git clone https://github.com/AnjanaSuresh01/handbook-generator
cd handbook-generator
pip install -e ".[dev]"

cp .env.example .env      # add a free Groq key: https://console.groq.com/keys
python app.py             # opens http://127.0.0.1:7860
```

Then, in the chat:

1. Upload one or more PDFs
2. Ask questions — answers cite the chunks they came from
3. Say **"generate a handbook on retrieval-augmented generation"**

## CLI

```bash
handbook plan paper.pdf --topic "Retrieval-Augmented Generation"
handbook ask paper.pdf --question "What problem does this solve?"
handbook generate a.pdf b.pdf --topic "RAG" -o handbook.md
```

`plan` prints the outline without writing anything, so you can sanity-check the
structure before spending tokens on a full run.

## Configuration

Everything is `.env`. Defaults work offline with no accounts.

**LLM** — any OpenAI-compatible endpoint:

| Provider | Base URL | Notes |
|---|---|---|
| **Groq** (default) | `https://api.groq.com/openai/v1` | Free tier generates a full handbook |
| xAI Grok | `https://api.x.ai/v1` | Three-line swap |
| Ollama | `http://localhost:11434/v1` | Fully local, no key, no network |

**Retrieval** — `STORAGE_BACKEND`:

- `local` (default) — BM25 index, no accounts, no embeddings, works offline
- `lightrag` — knowledge-graph retrieval via
  [LightRAG](https://github.com/HKUDS/LightRAG) (EMNLP 2025), which indexes
  ~60% cheaper than Microsoft GraphRAG at roughly half the query latency.
  `pip install -e ".[graph]"`

**PDF parsing** — `pypdf` by default; install `.[parse]` to use `liteparse`,
which runs OCR only on pages that need it and preserves tables as Markdown.

**Verification thresholds** — `VERIFY_MIN_GROUNDING`, `VERIFY_MAX_SIMILARITY`,
`VERIFY_MIN_WORD_RATIO`, `VERIFY_MAX_RETRIES`.

## Architecture

```
PDFs ──► ingest ──► store ──► outline ──► writer ──► verify ──► assemble ──► handbook.md
         extract    BM25 or   plan with   section    grounding   TOC,
         + chunk    LightRAG  budgets     by section repetition  citations,
                                          + retries  budget      quality report
```

| Module | Responsibility |
|---|---|
| `ingest.py` | PDF → cleaned text → overlapping chunks |
| `store.py` | Index and retrieve; BM25 or LightRAG behind one interface |
| `outline.py` | Plan sections, rebalance budgets to the target |
| `writer.py` | Generate each section, verify, retry on failure |
| `verify.py` | The quality gate |
| `assemble.py` | TOC, citations, quality report, export |
| `pipeline.py` | Session state; what the UI talks to |

## Design decisions

**Provider-agnostic LLM client.** A 20,000-word run is dozens of calls. You
develop against a free endpoint and switch to a paid one for the final run
without touching code.

**Runs with zero credentials.** The BM25 store means anyone can clone this and
index PDFs with no Supabase project and no API key. A reviewer who cannot run
your submission cannot evaluate it.

**Verification is deterministic.** Using an LLM to judge LLM output adds cost,
latency, and the judge's own error rate. Term overlap and n-gram similarity are
cheap, reproducible, and language-agnostic — and their limitations are stated
below rather than hidden.

## What this does not detect

- **Paraphrased invention.** A section that fabricates a plausible claim using
  only source vocabulary scores well on grounding. Catching that needs an
  entailment model.
- **Semantic repetition.** Two sections making the same point in entirely
  different words score low on n-gram similarity.
- **Factual correctness.** Grounding measures *vocabulary overlap with the
  sources*, not truth.
- **Threshold calibration.** The defaults are reasonable starting points, not
  validated constants. Tune them on your own documents.

## Development

```bash
pip install -e ".[dev]"
pytest -q          # 47 tests, no API key needed
ruff check src tests app.py
```

Tests use a stub LLM, so the suite runs offline and costs nothing.

## License

MIT © 2026 Anjana Suresh
