"""Thin OpenAI-compatible LLM client.

Deliberately provider-agnostic: xAI Grok, Groq, OpenAI and a local Ollama
server all speak this protocol, so switching providers is a .env change rather
than a code change. That matters here because generating a 20,000-word
handbook is expensive, and development wants a cheap or free endpoint even if
the final run uses Grok.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import LLMConfig

log = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when the provider cannot be reached or returns an unusable reply."""


@dataclass
class Usage:
    """Token accounting, aggregated across a whole run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, payload: dict) -> None:
        usage = payload.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        self.calls += 1


class LLMClient:
    """Minimal chat-completions client with retry and usage tracking."""

    def __init__(self, config: LLMConfig, *, max_retries: int = 3) -> None:
        self.config = config
        self.max_retries = max_retries
        self.usage = Usage()

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Send one chat completion and return the assistant text."""
        if not self.config.api_key:
            raise LLMError(
                "No LLM_API_KEY configured. Copy .env.example to .env and set a key, "
                "or point LLM_BASE_URL at a local Ollama server."
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        payload = self._post("/chat/completions", body)
        self.usage.add(payload)

        try:
            return payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape from provider: {payload!r}") from exc

    def complete_json(self, prompt: str, *, system: str | None = None, **kwargs) -> dict | list:
        """Complete and parse JSON, tolerating models that wrap it in fences."""
        raw = self.complete(prompt, system=system, **kwargs)
        return parse_json_block(raw)

    def _post(self, path: str, body: dict) -> dict:
        url = self.config.base_url.rstrip("/") + path
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in _RETRYABLE_STATUS:
                    detail = exc.read().decode("utf-8", "replace")[:500]
                    raise LLMError(f"Provider returned {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc

            backoff = 2**attempt
            log.warning("LLM call failed (attempt %d), retrying in %ss", attempt + 1, backoff)
            time.sleep(backoff)

        raise LLMError(f"LLM call failed after {self.max_retries} attempts: {last_error}")


def parse_json_block(text: str) -> dict | list:
    """Extract JSON from a model reply, tolerating ```json fences and prose.

    Models routinely wrap JSON in explanation despite instructions not to, so
    falling back to the outermost brace/bracket pair is the difference between
    a pipeline that works and one that fails on every third call.
    """
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:]).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise LLMError(f"Could not parse JSON from model reply: {text[:300]!r}")
