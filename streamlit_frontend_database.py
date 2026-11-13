"""
Frontend 3 – Persistent chat UI with conversation management.

Pairs with: langgraph_backend_database.py
Run with:   streamlit run streamlit_frontend_database.py

Adds a sidebar that lists past conversations loaded from SQLite.
Responses are streamed token-by-token.  New threads are created with
UUID4 IDs and persist across process restarts.

Supports Google Gemini, OpenAI, and Demo Mode.
"""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from llm_config import get_provider_display_name, is_demo_mode

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LangGraph Chatbot – Persistent",
    page_icon="💾",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Lazy backend import
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading chatbot…")
def _load_backend():
    try:
        from langgraph_backend_database import chatbot, retrieve_all_threads
        return chatbot, retrieve_all_threads, None
    except EnvironmentError as exc:
        return None, None, str(exc)
    except Exception as exc:
        return None, None, f"Failed to initialise backend: {exc}"


chatbot, retrieve_all_threads, _init_error = _load_backend()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_thread_id() -> str:
    return str(uuid.uuid4())


def _thread_title(thread_id: str, messages: list) -> str:
    """Use the first user message as the conversation title."""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = msg.content.strip().splitlines()[0]
            return text[:40] + "…" if len(text) > 40 else text
    return "New conversation"


def _load_history(thread_id: str) -> list[dict]:
    """Reload a conversation from the LangGraph checkpointer."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    formatted = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            formatted.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            formatted.append({"role": "assistant", "content": msg.content})
    return formatted


def _add_thread_to_sidebar(thread_id: str) -> None:
    """Ensure thread_id appears at the top of the sidebar list."""
    threads = st.session_state.chat_threads
    if thread_id in threads:
        threads.remove(thread_id)
    threads.insert(0, thread_id)


def _ai_text_stream(user_input: str, config: dict):
    """
    Generator that yields only text chunks from AIMessage objects.
    Filters out tool-call objects and empty strings.
    """
    for message_chunk, _metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        if isinstance(message_chunk, AIMessage) and message_chunk.content:
            yield message_chunk.content


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = _new_thread_id()

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "chat_threads" not in st.session_state:
    if retrieve_all_threads is not None:
        st.session_state.chat_threads = retrieve_all_threads()
    else:
        st.session_state.chat_threads = []

# Always ensure the current thread appears in the sidebar
if chatbot is not None:
    _add_thread_to_sidebar(st.session_state.thread_id)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("💾 LangGraph Chatbot")
    st.caption("Persistent · SQLite · Streaming")

    if is_demo_mode():
        st.info("🎭 Demo Mode", icon="ℹ️")
    else:
        st.caption(f"Provider: {get_provider_display_name()}")

    st.divider()

    if chatbot is not None:
        if st.button("➕  New Chat", use_container_width=True):
            st.session_state.thread_id = _new_thread_id()
            st.session_state.message_history = []
            _add_thread_to_sidebar(st.session_state.thread_id)
            st.rerun()

        st.subheader("Conversations")
        for tid in st.session_state.chat_threads:
            try:
                messages_raw = chatbot.get_state(
                    config={"configurable": {"thread_id": tid}}
                ).values.get("messages", [])
                label = _thread_title(tid, messages_raw)
            except Exception:
                label = str(tid)[:8] + "…"

            is_active = tid == st.session_state.thread_id
            btn_label = f"**{label}**" if is_active else label

            if st.button(btn_label, key=f"thread-btn-{tid}", use_container_width=True):
                st.session_state.thread_id = tid
                st.session_state.message_history = _load_history(tid)
                st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("💬 Chat")
st.caption(f"Thread: `{st.session_state.thread_id}`")

if is_demo_mode():
    st.info(
        "🎭 **Demo Mode active** — responses are simulated. "
        "Add an API key to `.env` for live AI responses.",
        icon="ℹ️",
    )

# Show initialisation error
if _init_error:
    st.error(f"⚠️ Configuration error:\n\n{_init_error}")
    st.stop()

st.divider()

# Render existing history
for msg in st.session_state.message_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your message…")

if user_input:
    # Append and render user message immediately
    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Stream the assistant reply
    with st.chat_message("assistant"):
        try:
            ai_message = st.write_stream(_ai_text_stream(user_input, config))
        except EnvironmentError as exc:
            ai_message = f"⚠️ Configuration error: {exc}"
            st.markdown(ai_message)
        except Exception as exc:
            ai_message = f"⚠️ Error: {exc}"
            st.markdown(ai_message)

    st.session_state.message_history.append(
        {"role": "assistant", "content": ai_message or ""}
    )
