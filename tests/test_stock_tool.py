"""
Tests for the get_stock_price tool (langgraph_backend_tool.py).

All HTTP calls are mocked — no real network requests are made and no
API key is required to run this test suite.

Uses DEMO_MODE=true so the backend imports succeed without a real LLM key.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixture: ensure DEMO_MODE is set so the backend module loads cleanly
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def demo_mode(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    # Reload backend to pick up demo mode
    import importlib
    import langgraph_backend_tool
    importlib.reload(langgraph_backend_tool)
    yield


def _get_tool_func():
    from langgraph_backend_tool import get_stock_price as _tool
    return _tool.func


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_success(monkeypatch):
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "Global Quote": {
            "01. symbol": "AAPL",
            "05. price": "189.30",
        }
    }
    fake_response.raise_for_status = MagicMock()

    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "fake-key")

    with patch("langgraph_backend_tool.requests.get", return_value=fake_response):
        result = _get_tool_func()(symbol="AAPL")

    assert "Global Quote" in result
    assert result["Global Quote"]["01. symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    result = _get_tool_func()(symbol="AAPL")
    assert "error" in result
    assert "ALPHA_VANTAGE_API_KEY" in result["error"]


def test_empty_global_quote(monkeypatch):
    fake_response = MagicMock()
    fake_response.json.return_value = {"Global Quote": {}}
    fake_response.raise_for_status = MagicMock()

    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "fake-key")

    with patch("langgraph_backend_tool.requests.get", return_value=fake_response):
        result = _get_tool_func()(symbol="INVALID")

    assert "error" in result


def test_network_timeout(monkeypatch):
    import requests as req

    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "fake-key")

    with patch("langgraph_backend_tool.requests.get", side_effect=req.exceptions.Timeout):
        result = _get_tool_func()(symbol="AAPL")

    assert "error" in result
    assert "timed out" in result["error"].lower()


def test_network_error(monkeypatch):
    import requests as req

    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "fake-key")

    with patch(
        "langgraph_backend_tool.requests.get",
        side_effect=req.exceptions.ConnectionError("unreachable"),
    ):
        result = _get_tool_func()(symbol="AAPL")

    assert "error" in result


def test_rate_limit_note(monkeypatch):
    """Alpha Vantage returns a 'Note' key when the rate limit is hit."""
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "Note": "Thank you for using Alpha Vantage! Our standard API call limit is 25..."
    }
    fake_response.raise_for_status = MagicMock()

    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "fake-key")

    with patch("langgraph_backend_tool.requests.get", return_value=fake_response):
        result = _get_tool_func()(symbol="AAPL")

    assert "error" in result
    assert "rate limit" in result["error"].lower()
