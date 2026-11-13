"""
Tests for the centralized LLM configuration module (llm_config.py).

Covers:
  • Provider auto-detection from environment variables
  • Demo mode activation
  • Missing API key error handling
  • Invalid provider name rejection
  • DemoLLM stub behavior
  • Provider display name helper

No real API calls are made — all LLM constructors are tested with
environment mocking only.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: import fresh to avoid module-level caching
# ---------------------------------------------------------------------------

def _fresh_import():
    """Return a freshly-imported llm_config module."""
    import importlib
    import llm_config
    importlib.reload(llm_config)
    return llm_config


# ---------------------------------------------------------------------------
# Demo mode detection
# ---------------------------------------------------------------------------

class TestIsDemoMode:
    def test_demo_true(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "true")
        cfg = _fresh_import()
        assert cfg.is_demo_mode() is True

    def test_demo_false(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "false")
        cfg = _fresh_import()
        assert cfg.is_demo_mode() is False

    def test_demo_uppercase(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "TRUE")
        cfg = _fresh_import()
        assert cfg.is_demo_mode() is True

    def test_demo_1(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "1")
        cfg = _fresh_import()
        assert cfg.is_demo_mode() is True

    def test_demo_absent(self, monkeypatch):
        monkeypatch.delenv("DEMO_MODE", raising=False)
        cfg = _fresh_import()
        assert cfg.is_demo_mode() is False


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

class TestGetProvider:
    def test_demo_mode_overrides_provider(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "true")
        monkeypatch.setenv("AI_PROVIDER", "openai")
        cfg = _fresh_import()
        assert cfg.get_provider() == "demo"

    def test_explicit_gemini(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "false")
        monkeypatch.setenv("AI_PROVIDER", "gemini")
        cfg = _fresh_import()
        assert cfg.get_provider() == "gemini"

    def test_explicit_openai(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "false")
        monkeypatch.setenv("AI_PROVIDER", "openai")
        cfg = _fresh_import()
        assert cfg.get_provider() == "openai"

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "false")
        monkeypatch.setenv("AI_PROVIDER", "GEMINI")
        cfg = _fresh_import()
        assert cfg.get_provider() == "gemini"

    def test_auto_detects_google_key(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "false")
        monkeypatch.delenv("AI_PROVIDER", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = _fresh_import()
        assert cfg.get_provider() == "gemini"

    def test_auto_detects_openai_key(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "false")
        monkeypatch.delenv("AI_PROVIDER", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
        cfg = _fresh_import()
        assert cfg.get_provider() == "openai"


# ---------------------------------------------------------------------------
# get_llm — demo mode
# ---------------------------------------------------------------------------

class TestGetLlmDemo:
    def test_returns_demo_llm(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "true")
        cfg = _fresh_import()
        llm = cfg.get_llm()
        assert isinstance(llm, cfg.DemoLLM)

    def test_demo_llm_invoke_returns_ai_message(self, monkeypatch):
        from langchain_core.messages import AIMessage, HumanMessage
        monkeypatch.setenv("DEMO_MODE", "true")
        cfg = _fresh_import()
        llm = cfg.get_llm()
        result = llm.invoke([HumanMessage(content="Hello")])
        assert isinstance(result, AIMessage)
        assert "[DEMO MODE]" in result.content or "demo" in result.content.lower()

    def test_demo_llm_bind_tools_returns_self(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "true")
        cfg = _fresh_import()
        llm = cfg.get_llm()
        bound = llm.bind_tools([])
        assert bound is llm

    def test_demo_llm_stream(self, monkeypatch):
        from langchain_core.messages import AIMessage, HumanMessage
        monkeypatch.setenv("DEMO_MODE", "true")
        cfg = _fresh_import()
        llm = cfg.get_llm()
        chunks = list(llm.stream([HumanMessage(content="hi")]))
        assert len(chunks) >= 1
        assert isinstance(chunks[0], AIMessage)


# ---------------------------------------------------------------------------
# get_llm — missing API key errors
# ---------------------------------------------------------------------------

class TestGetLlmMissingKey:
    def test_missing_google_key_raises(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "false")
        monkeypatch.setenv("AI_PROVIDER", "gemini")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        cfg = _fresh_import()
        with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
            cfg.get_llm("gemini")

    def test_missing_openai_key_raises(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "false")
        monkeypatch.setenv("AI_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = _fresh_import()
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            cfg.get_llm("openai")

    def test_invalid_provider_raises(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "false")
        cfg = _fresh_import()
        with pytest.raises(ValueError, match="Unsupported provider"):
            cfg.get_llm("anthropic")


# ---------------------------------------------------------------------------
# get_llm — with keys present (mocked constructors)
# ---------------------------------------------------------------------------

class TestGetLlmWithKeys:
    def test_gemini_constructed_with_key(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "false")
        monkeypatch.setenv("AI_PROVIDER", "gemini")
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-gemini-key")

        mock_llm = MagicMock()
        with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm) as mock_cls:
            cfg = _fresh_import()
            result = cfg.get_llm("gemini")
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args.kwargs
            # Key must be passed; must NOT be leaked in assertions
            assert "google_api_key" in call_kwargs

    def test_openai_constructed_with_key(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "false")
        monkeypatch.setenv("AI_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")

        mock_llm = MagicMock()
        with patch("langchain_openai.ChatOpenAI", return_value=mock_llm) as mock_cls:
            cfg = _fresh_import()
            result = cfg.get_llm("openai")
            mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# Display name helper
# ---------------------------------------------------------------------------

class TestGetProviderDisplayName:
    def test_demo_label(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "true")
        cfg = _fresh_import()
        name = cfg.get_provider_display_name()
        assert "demo" in name.lower() or "simulated" in name.lower()

    def test_gemini_label(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "false")
        monkeypatch.setenv("AI_PROVIDER", "gemini")
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        cfg = _fresh_import()
        name = cfg.get_provider_display_name()
        assert "gemini" in name.lower()

    def test_openai_label(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "false")
        monkeypatch.setenv("AI_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        cfg = _fresh_import()
        name = cfg.get_provider_display_name()
        assert "openai" in name.lower()
