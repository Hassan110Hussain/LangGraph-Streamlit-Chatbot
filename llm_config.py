"""
Centralized LLM configuration module.

Supports two AI providers:
  - Google Gemini  (AI_PROVIDER=gemini,  key: GOOGLE_API_KEY)
  - OpenAI         (AI_PROVIDER=openai,  key: OPENAI_API_KEY)

Demo mode (DEMO_MODE=true) bypasses all provider initialization
and returns a DemoLLM stub that generates clearly labeled simulated responses.

Usage
-----
    from llm_config import get_llm, get_provider, is_demo_mode

    llm = get_llm()                   # ready-to-use ChatModel
    provider = get_provider()         # "gemini" | "openai" | "demo"
    demo = is_demo_mode()             # bool
"""

from __future__ import annotations

import os
from typing import Any, Iterator, Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

SUPPORTED_PROVIDERS = ("gemini", "openai")


# ---------------------------------------------------------------------------
# Demo LLM stub
# ---------------------------------------------------------------------------
class DemoLLM:
    """
    A drop-in stub that mimics the LangChain ChatModel interface.

    Returns clearly labelled simulated responses — never pretends to be
    a real AI model.  No API key required.
    """

    # Canned responses keyed on keywords in the user message
    _RESPONSES: dict[str, str] = {
        "hello": (
            "👋 **[DEMO MODE]** Hello! I'm a simulated assistant running in demo mode. "
            "No real AI model is active. Configure GOOGLE_API_KEY or OPENAI_API_KEY "
            "in your `.env` file to enable live responses."
        ),
        "calculator": (
            "🧮 **[DEMO MODE]** I can call the calculator tool! "
            "For example: 42 × 7 = 294. "
            "*(In live mode this would be computed by the actual tool.)*"
        ),
        "stock": (
            "📈 **[DEMO MODE]** Stock lookup is available when ALPHA_VANTAGE_API_KEY "
            "is configured. Here I'd return a simulated quote: DEMO at $123.45 (+1.2%)."
        ),
        "search": (
            "🔍 **[DEMO MODE]** Web search via DuckDuckGo is active in live mode. "
            "I'd search the web and summarise results for you."
        ),
        "thread": (
            "🗂️ **[DEMO MODE]** Thread switching is working! "
            "You can create new conversations and switch between them — "
            "this feature uses SQLite persistence under the hood."
        ),
        "history": (
            "📜 **[DEMO MODE]** Conversation history is stored in SQLite. "
            "Each thread keeps its own message log that survives app restarts."
        ),
    }
    _DEFAULT_RESPONSE = (
        "🤖 **[DEMO MODE — Simulated Response]**\n\n"
        "This is a placeholder response generated without a real AI model. "
        "To get actual AI responses, add your API key to `.env`:\n\n"
        "- **Google Gemini**: set `GOOGLE_API_KEY` and `AI_PROVIDER=gemini`\n"
        "- **OpenAI**: set `OPENAI_API_KEY` and `AI_PROVIDER=openai`\n\n"
        "The app is fully functional in demo mode — you can explore threading, "
        "persistence, tool routing, and the UI without any credentials."
    )

    def _pick_response(self, messages: list) -> str:
        """Return a canned response based on the last user message."""
        last_text = ""
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                last_text = content.lower()
                break

        for keyword, reply in self._RESPONSES.items():
            if keyword in last_text:
                return reply
        return self._DEFAULT_RESPONSE

    # ------------------------------------------------------------------
    # LangChain-compatible interface
    # ------------------------------------------------------------------

    def invoke(self, messages: list, **kwargs: Any):
        from langchain_core.messages import AIMessage
        return AIMessage(content=self._pick_response(messages))

    async def ainvoke(self, messages: list, **kwargs: Any):
        from langchain_core.messages import AIMessage
        return AIMessage(content=self._pick_response(messages))

    def stream(self, messages: list, **kwargs: Any) -> Iterator:
        from langchain_core.messages import AIMessage
        # Yield the full response as a single chunk (simulated stream)
        yield AIMessage(content=self._pick_response(messages))

    async def astream(self, messages: list, **kwargs: Any):
        from langchain_core.messages import AIMessage
        yield AIMessage(content=self._pick_response(messages))

    def bind_tools(self, tools: list, **kwargs: Any) -> "DemoLLM":
        """Return self — demo mode does not use real tool calling."""
        return self

    def __repr__(self) -> str:
        return "DemoLLM(simulated)"


# ---------------------------------------------------------------------------
# Provider / mode detection
# ---------------------------------------------------------------------------

def is_demo_mode() -> bool:
    """Return True if DEMO_MODE=true in the environment."""
    return os.getenv("DEMO_MODE", "false").strip().lower() in ("true", "1", "yes")


def get_provider() -> str:
    """
    Return the active provider name: 'gemini', 'openai', or 'demo'.

    Resolution order:
      1. DEMO_MODE=true  → 'demo'
      2. AI_PROVIDER env var (case-insensitive)
      3. Fall back to 'gemini' if GOOGLE_API_KEY is set, else 'openai'
    """
    if is_demo_mode():
        return "demo"

    raw = os.getenv("AI_PROVIDER", "").strip().lower()
    if raw in SUPPORTED_PROVIDERS:
        return raw

    # Auto-detect from which key is present
    if os.getenv("GOOGLE_API_KEY", "").strip():
        return "gemini"
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"

    # Default to gemini (user will see a clear error on first use)
    return "gemini"


def get_llm(provider: Optional[str] = None) -> Any:
    """
    Build and return a ready-to-use LangChain ChatModel.

    Parameters
    ----------
    provider : str | None
        'gemini', 'openai', or 'demo'. If None, resolved from environment.

    Returns
    -------
    A LangChain BaseChatModel (or DemoLLM stub in demo mode).

    Raises
    ------
    EnvironmentError
        When the required API key for the selected provider is missing.
    ValueError
        When an unsupported provider name is supplied.
    """
    if provider is None:
        provider = get_provider()

    provider = provider.strip().lower()

    if provider == "demo":
        return DemoLLM()

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. "
                "Add it to your .env file or set AI_PROVIDER=openai to use OpenAI instead. "
                "Set DEMO_MODE=true to explore without any API key."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "langchain-google-genai is not installed. "
                "Run: pip install langchain-google-genai"
            ) from exc

        model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            convert_system_message_to_human=True,
        )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or set AI_PROVIDER=gemini to use Gemini instead. "
                "Set DEMO_MODE=true to explore without any API key."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is not installed. "
                "Run: pip install langchain-openai"
            ) from exc

        model_name = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
        )

    raise ValueError(
        f"Unsupported provider '{provider}'. "
        f"Choose from: {', '.join(SUPPORTED_PROVIDERS)} or set DEMO_MODE=true."
    )


def get_provider_display_name() -> str:
    """Return a human-readable provider label for display in the UI."""
    provider = get_provider()
    names = {
        "gemini": f"Google Gemini ({os.getenv('GEMINI_MODEL', DEFAULT_GEMINI_MODEL)})",
        "openai": f"OpenAI ({os.getenv('OPENAI_MODEL', DEFAULT_OPENAI_MODEL)})",
        "demo": "Demo Mode (simulated responses)",
    }
    return names.get(provider, provider)
