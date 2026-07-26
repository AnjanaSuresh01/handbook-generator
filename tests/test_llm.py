from handbook.config import LLMConfig
from handbook.llm import LLMClient, _explain


class TestHeaders:
    def test_sets_an_explicit_user_agent(self):
        """Regression guard.

        Providers behind Cloudflare (Groq among them) reject the default
        "Python-urllib/x.y" User-Agent with 403 error 1010 before the request
        reaches the API, so the header must never be left to urllib.
        """
        headers = LLMClient(LLMConfig()).build_headers()
        agent = headers.get("User-Agent", "")
        assert agent
        assert "urllib" not in agent.lower()

    def test_sends_bearer_token(self):
        headers = LLMClient(LLMConfig()).build_headers()
        assert headers["Authorization"].startswith("Bearer ")

    def test_sends_json_content_type(self):
        assert LLMClient(LLMConfig()).build_headers()["Content-Type"] == "application/json"


class TestErrorMessages:
    def _explain(self, status, detail="oops"):
        return _explain(status, detail, LLMConfig())

    def test_401_points_at_the_env_file(self):
        assert ".env" in self._explain(401)

    def test_403_mentions_bot_protection(self):
        assert "1010" in self._explain(403)

    def test_404_names_the_configured_model(self):
        assert LLMConfig().model in self._explain(404)

    def test_429_mentions_rate_limit(self):
        assert "Rate limit" in self._explain(429)

    def test_unknown_status_still_includes_the_detail(self):
        message = self._explain(500, "internal boom")
        assert "500" in message
        assert "internal boom" in message


class TestRateLimitHandling:
    def test_429_is_not_in_the_generic_retry_set(self):
        """429 must take the wait-and-retry path, not the 3-strikes path."""
        from handbook.llm import _RATE_LIMITED, _RETRYABLE_STATUS

        assert _RATE_LIMITED == 429
        assert 429 not in _RETRYABLE_STATUS

    def test_retry_after_header_is_honoured(self):
        import urllib.error

        from handbook.llm import _retry_after

        exc = urllib.error.HTTPError("u", 429, "x", {"Retry-After": "17"}, None)
        assert _retry_after(exc) == 17.0

    def test_retry_after_is_capped(self):
        import urllib.error

        from handbook.llm import _retry_after

        exc = urllib.error.HTTPError("u", 429, "x", {"Retry-After": "99999"}, None)
        assert _retry_after(exc) == 300.0

    def test_missing_or_malformed_retry_after_returns_none(self):
        import urllib.error

        from handbook.llm import _retry_after

        assert _retry_after(urllib.error.HTTPError("u", 429, "x", {}, None)) is None
        assert (
            _retry_after(urllib.error.HTTPError("u", 429, "x", {"Retry-After": "soon"}, None))
            is None
        )

    def test_throttle_waits_between_calls(self):
        import time

        client = LLMClient(LLMConfig(), min_interval=0.25)
        client._last_call_at = time.monotonic()
        started = time.monotonic()
        client._throttle()
        assert time.monotonic() - started >= 0.2

    def test_throttle_is_skipped_when_disabled(self):
        import time

        client = LLMClient(LLMConfig(), min_interval=0)
        started = time.monotonic()
        client._throttle()
        assert time.monotonic() - started < 0.05

    def test_429_message_suggests_a_smaller_target(self):
        assert "HANDBOOK_TARGET_WORDS" in _explain(429, "limit", LLMConfig())
