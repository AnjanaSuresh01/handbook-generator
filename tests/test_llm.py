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
