"""Runtime configuration, loaded from environment with usable defaults.

Every knob has a default that works offline, so the app starts without any
credentials and degrades to local models rather than crashing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: .env support is convenient but not required
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _env(key: str, default: str) -> str:
    value = os.environ.get(key, "").strip()
    return value or default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class LLMConfig:
    """Provider settings.

    Defaults to Groq because its free tier can generate a full handbook; xAI
    Grok and a local Ollama server are drop-in replacements via .env.
    """

    base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    )
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "llama-3.3-70b-versatile"))
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    timeout: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 180.0))
    #: Minimum seconds between calls. Free tiers cap requests per minute, and
    #: pacing costs less total time than repeatedly tripping the limit.
    min_interval: float = field(default_factory=lambda: _env_float("LLM_MIN_INTERVAL", 2.0))

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class StorageConfig:
    backend: str = field(default_factory=lambda: _env("STORAGE_BACKEND", "local"))
    working_dir: Path = field(default_factory=lambda: Path(_env("WORKING_DIR", "./storage")))
    supabase_url: str = field(default_factory=lambda: _env("SUPABASE_URL", ""))
    supabase_key: str = field(default_factory=lambda: _env("SUPABASE_KEY", ""))
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "local"))
    embedding_model: str = field(
        default_factory=lambda: _env(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    )


@dataclass(frozen=True)
class GenerationConfig:
    """Targets for the plan-then-write pipeline.

    ``target_words`` is the contract with the user; the planner distributes it
    across sections as a per-section budget so no single call has to produce
    more than a few thousand words.
    """

    target_words: int = field(default_factory=lambda: _env_int("HANDBOOK_TARGET_WORDS", 20_000))
    min_sections: int = field(default_factory=lambda: _env_int("HANDBOOK_MIN_SECTIONS", 12))
    max_sections: int = field(default_factory=lambda: _env_int("HANDBOOK_MAX_SECTIONS", 24))
    max_concurrency: int = field(default_factory=lambda: _env_int("HANDBOOK_MAX_CONCURRENCY", 4))


@dataclass(frozen=True)
class VerifyConfig:
    """Thresholds for the quality gate between generation and assembly."""

    max_retries: int = field(default_factory=lambda: _env_int("VERIFY_MAX_RETRIES", 2))
    min_grounding: float = field(default_factory=lambda: _env_float("VERIFY_MIN_GROUNDING", 0.35))
    max_similarity: float = field(default_factory=lambda: _env_float("VERIFY_MAX_SIMILARITY", 0.75))
    min_word_ratio: float = field(default_factory=lambda: _env_float("VERIFY_MIN_WORD_RATIO", 0.70))


@dataclass(frozen=True)
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)


def load_settings() -> Settings:
    return Settings()
