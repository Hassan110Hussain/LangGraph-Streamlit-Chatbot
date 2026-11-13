"""
Frontend 2 – Streaming chat UI (in-memory backend).

Pairs with: langgraph_backend.py
Run with:   streamlit run streamlit_frontend_streaming.py

Demonstrates token-by-token streaming using st.write_stream.
Filters out tool-call chunks and empty content so only real AI text
reaches the stream.

Supports Google Gemini, OpenAI, and Demo Mode.
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from llm_config import get_provider_display_name, is_demo_mode

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LangGraph Chatbot – Streaming",
    page_icon="⚡",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Lazy backend import — catches missing API key at runtime with a clear error
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading chatbot…")
def _load_backend():
    try:
        from langgraph_backend import chatbot
        return chatbot, None
    except EnvironmentError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"Failed to initialise backend: {exc}"


chatbot, _init_error = _load_backend()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
THREAD_ID = "streaming-thread"
CONFIG = {"configurable": {"thread_id": THREAD_ID}}

if "message_history" not in st.session_state:
    st.session_state.message_history = []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ai_text_stream(user_input: str):
    """
    Generator that yields only the text content of AIMessage chunks.
    Skips tool-call chunks and empty strings so st.write_stream never
    receives None or blank tokens.
    """
    for message_chunk, _metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=CONFIG,
        stream_mode="messages",
    ):
        if isinstance(message_chunk, AIMessage) and message_chunk.content:
            yield message_chunk.content


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("⚡ LangGraph Chatbot — Streaming")
st.caption("Backend 1 · In-memory · Streaming responses · No persistence")

# Provider badge / demo banner
if is_demo_mode():
    st.info(
        "🎭 **Demo Mode active** — responses are simulated. "
        "Add an API key to `.env` for live AI responses.",
        icon="ℹ️",
    )
else:
    st.caption(f"Provider: {get_provider_display_name()}")

# Show initialisation error
if _init_error:
    st.error(f"⚠️ Configuration error:\n\n{_init_error}")
    st.stop()

st.divider()

# Render chat history
for msg in st.session_state.message_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask me anything…")

if user_input:
    # Show user message immediately
    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream assistant reply
    with st.chat_message("assistant"):
        try:
            ai_message = st.write_stream(_ai_text_stream(user_input))
        except EnvironmentError as exc:
            ai_message = f"⚠️ Configuration error: {exc}"
            st.markdown(ai_message)
        except Exception as exc:
            ai_message = f"⚠️ Streaming error: {exc}"
            st.markdown(ai_message)

    st.session_state.message_history.append(
        {"role": "assistant", "content": ai_message or ""}
    )
