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

_RETRYABLE_STATUS = {408, 500, 502, 503, 504}
_RATE_LIMITED = 429

# Several providers (Groq among them) sit behind Cloudflare, which rejects the
# default "Python-urllib/x.y" User-Agent as bot traffic with 403 error 1010
# before the request ever reaches the API. Any explicit value avoids this.
_USER_AGENT = "handbook-generator/0.1 (+https://github.com/AnjanaSuresh01/handbook-generator)"


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
    """Minimal chat-completions client with retry, throttling and usage tracking.

    Rate limits are handled separately from transient errors. A 429 is not a
    failure to retry a few times and give up on -- it is an instruction to wait,
    usually with a Retry-After header saying exactly how long. Free tiers reset
    per minute, so a handbook run has to be prepared to sit still for a while
    rather than burn its retry budget in eight seconds.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        max_retries: int = 3,
        max_rate_limit_waits: int = 12,
        min_interval: float | None = None,
    ) -> None:
        self.config = config
        self.max_retries = max_retries
        self.max_rate_limit_waits = max_rate_limit_waits
        self.min_interval = config.min_interval if min_interval is None else min_interval
        self.usage = Usage()
        self._last_call_at = 0.0

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

    def build_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            "User-Agent": _USER_AGENT,
        }

    def _throttle(self) -> None:
        """Keep a minimum gap between calls to stay under requests-per-minute caps."""
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_call_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _post(self, path: str, body: dict) -> dict:
        url = self.config.base_url.rstrip("/") + path
        data = json.dumps(body).encode("utf-8")
        headers = self.build_headers()

        last_error: Exception | None = None
        attempts = 0
        rate_limit_waits = 0

        while True:
            self._throttle()
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    self._last_call_at = time.monotonic()
                    return json.loads(response.read().decode("utf-8"))

            except urllib.error.HTTPError as exc:
                self._last_call_at = time.monotonic()
                last_error = exc

                if exc.code == _RATE_LIMITED:
                    rate_limit_waits += 1
                    if rate_limit_waits > self.max_rate_limit_waits:
                        detail = exc.read().decode("utf-8", "replace")[:500]
                        raise LLMError(_explain(exc.code, detail, self.config)) from exc
                    delay = _retry_after(exc) or min(60.0, 10.0 * rate_limit_waits)
                    log.warning(
                        "Rate limited by the provider; waiting %.0fs (pause %d of %d)",
                        delay,
                        rate_limit_waits,
                        self.max_rate_limit_waits,
                    )
                    time.sleep(delay)
                    continue

                if exc.code not in _RETRYABLE_STATUS:
                    detail = exc.read().decode("utf-8", "replace")[:500]
                    raise LLMError(_explain(exc.code, detail, self.config)) from exc

            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                self._last_call_at = time.monotonic()
                last_error = exc

            attempts += 1
            if attempts >= self.max_retries:
                break
            backoff = 2**attempts
            log.warning("LLM call failed (attempt %d), retrying in %ss", attempts, backoff)
            time.sleep(backoff)

        raise LLMError(f"LLM call failed after {self.max_retries} attempts: {last_error}")


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    """Seconds to wait, from the provider's Retry-After header.

    Providers say exactly how long to wait; guessing when we have been told is
    both slower and ruder. Values are capped so a malformed header cannot hang
    a run for hours.
    """
    raw = (exc.headers.get("Retry-After") or "").strip() if exc.headers else ""
    if not raw:
        return None
    try:
        return max(1.0, min(float(raw), 300.0))
    except ValueError:
        return None


def _explain(status: int, detail: str, config: LLMConfig) -> str:
    """Turn a raw HTTP failure into something a user can act on."""
    hints = {
        401: "The API key was rejected. Check LLM_API_KEY in your .env file.",
        403: (
            "Access refused. If this mentions error code 1010 the provider's "
            "bot protection blocked the request; if it mentions the model, your "
            f"key may not have access to '{config.model}'."
        ),
        404: (
            f"Not found. Check LLM_BASE_URL ({config.base_url}) and that the "
            f"model '{config.model}' still exists with this provider."
        ),
        413: "The request was too large. Lower the retrieval context size.",
        429: (
            "Rate limit or daily quota exhausted, and waiting did not clear it. "
            "On a free tier, generate a shorter handbook first "
            "(set HANDBOOK_TARGET_WORDS=3000 in .env), raise LLM_MIN_INTERVAL to "
            "slow the request rate, or continue tomorrow when the daily quota resets."
        ),
    }
    hint = hints.get(status, "")
    return f"Provider returned {status}: {detail}" + (f"\n\n{hint}" if hint else "")


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
