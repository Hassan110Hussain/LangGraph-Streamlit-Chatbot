"""
Frontend 1 – Minimal chat UI (in-memory backend).

Pairs with: langgraph_backend.py
Run with:   streamlit run streamlit_frontend.py

The simplest possible Streamlit chatbot — single hardcoded thread,
no sidebar, no persistence between browser refreshes.  Great for
understanding the basic request/response loop.

Supports Google Gemini, OpenAI, and Demo Mode.
"""

import streamlit as st
from langchain_core.messages import HumanMessage

from llm_config import get_provider_display_name, is_demo_mode

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LangGraph Chatbot – Basic",
    page_icon="🤖",
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
THREAD_ID = "default-thread"
CONFIG = {"configurable": {"thread_id": THREAD_ID}}

if "message_history" not in st.session_state:
    st.session_state.message_history = []

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🤖 LangGraph Chatbot")
st.caption("Backend 1 · In-memory · No tools · No persistence")

# Provider badge
provider_label = get_provider_display_name()
if is_demo_mode():
    st.info(
        "🎭 **Demo Mode active** — responses are simulated. "
        "Add an API key to `.env` for live AI responses.",
        icon="ℹ️",
    )
else:
    st.caption(f"Provider: {provider_label}")

# Show initialisation error (missing key, bad import, etc.)
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
    # Show the user message immediately
    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call the backend and render the reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = chatbot.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                )
                ai_message = response["messages"][-1].content
            except EnvironmentError as exc:
                ai_message = f"⚠️ Configuration error: {exc}"
            except Exception as exc:
                ai_message = f"⚠️ An error occurred: {exc}"

        st.markdown(ai_message)

    st.session_state.message_history.append(
        {"role": "assistant", "content": ai_message}
    )
