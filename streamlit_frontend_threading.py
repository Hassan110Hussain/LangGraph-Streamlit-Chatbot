"""
Frontend 4 – Full-featured chat UI with tools + live status indicators.

Pairs with: langgraph_backend_tool.py
Run with:   streamlit run streamlit_frontend_threading.py

The most feature-rich frontend:
  • Sidebar with conversation history (titles derived from first user message)
  • st.status() shows which tool the agent is currently executing
  • Only AIMessage text is rendered; tool calls/results are hidden from the user
  • Thread state is fully persisted via SQLite

Supports Google Gemini, OpenAI, and Demo Mode.
"""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llm_config import get_provider_display_name, is_demo_mode

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LangGraph Chatbot – Tools",
    page_icon="🛠️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Lazy backend import
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading chatbot…")
def _load_backend():
    try:
        from langgraph_backend_tool import chatbot, retrieve_all_threads
        return chatbot, retrieve_all_threads, None
    except EnvironmentError as exc:
        return None, None, str(exc)
    except Exception as exc:
        return None, None, f"Failed to initialise backend: {exc}"


chatbot, retrieve_all_threads, _init_error = _load_backend()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TOPIC = "New conversation"
MAX_TOPIC_LEN = 55


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_thread_id() -> str:
    return str(uuid.uuid4())


def _truncate(text: str, max_len: int = MAX_TOPIC_LEN) -> str:
    text = text.strip()
    return text[: max_len - 3] + "…" if len(text) > max_len else text


def _derive_topic(messages: list) -> str | None:
    """Return the first non-empty human message, truncated."""
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content.strip():
            return _truncate(msg.content.strip())
    return None


def _load_history(thread_id: str) -> list[dict]:
    """Return display-ready messages for a thread (skips tool messages)."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    result = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            result.append({"role": "assistant", "content": msg.content})
        # ToolMessages are intentionally excluded from the display layer
    return result


def _extract_tool_name(chunk, metadata: dict) -> str | None:
    """Best-effort extraction of the tool name from a streaming chunk."""
    if isinstance(chunk, ToolMessage):
        return chunk.name

    tool_calls = getattr(chunk, "tool_calls", None)
    if tool_calls:
        first = tool_calls[0] or {}
        if "name" in first:
            return first["name"]
        func = first.get("function")
        if func:
            return func.get("name")

    return metadata.get("tool_name") or metadata.get("langgraph_node")


def _add_thread(thread_id: str) -> None:
    """Ensure thread_id is at the top of the sidebar list."""
    threads = st.session_state.chat_threads
    if thread_id in threads:
        threads.remove(thread_id)
    threads.insert(0, thread_id)


def _reset_chat() -> None:
    tid = _new_thread_id()
    st.session_state.thread_id = tid
    st.session_state.message_history = []
    st.session_state.thread_topics[tid] = DEFAULT_TOPIC
    _add_thread(tid)


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

if "thread_topics" not in st.session_state:
    st.session_state.thread_topics = {}

if chatbot is not None:
    _add_thread(st.session_state.thread_id)

st.session_state.thread_topics.setdefault(
    st.session_state.thread_id, DEFAULT_TOPIC
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛠️ LangGraph Chatbot")
    st.caption("Tools · Persistent · Streaming")

    if is_demo_mode():
        st.info("🎭 Demo Mode", icon="ℹ️")
    else:
        st.caption(f"Provider: {get_provider_display_name()}")

    st.divider()

    if chatbot is not None:
        if st.button("➕  New Chat", use_container_width=True):
            _reset_chat()
            st.rerun()

        st.subheader("Conversations")
        for tid in st.session_state.chat_threads:
            topic = st.session_state.thread_topics.get(tid, DEFAULT_TOPIC)
            is_active = tid == st.session_state.thread_id
            label = f"**{topic}**" if is_active else topic

            if st.button(label, key=f"thread-{tid}", use_container_width=True):
                st.session_state.thread_id = tid
                msgs = _load_history(tid)
                st.session_state.message_history = msgs
                raw_state = chatbot.get_state(
                    config={"configurable": {"thread_id": tid}}
                ).values.get("messages", [])
                derived = _derive_topic(raw_state)
                if derived:
                    st.session_state.thread_topics[tid] = derived
                st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("💬 Chat with Tools")
st.caption(f"Thread: `{st.session_state.thread_id}`")

if is_demo_mode():
    st.info(
        "🎭 **Demo Mode active** — responses are simulated. "
        "Tools (calculator, search, stocks) are available in live mode. "
        "Add an API key to `.env` to enable full functionality.",
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
user_input = st.chat_input(
    "Ask me anything — I can search the web, check stocks, and calculate…"
)

if user_input:
    # Render user message
    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Update thread topic from first user message
    if st.session_state.thread_topics.get(st.session_state.thread_id) == DEFAULT_TOPIC:
        st.session_state.thread_topics[st.session_state.thread_id] = _truncate(user_input)

    CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Stream with live tool status
    assistant_container = st.chat_message("assistant")
    status_box = assistant_container.status("Thinking…", state="running")

    def _ai_stream():
        current_tool: str | None = None
        try:
            for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                tool_name = _extract_tool_name(chunk, metadata or {})
                if tool_name and tool_name != current_tool:
                    current_tool = tool_name
                    status_box.update(
                        label=f"🔧 Using tool: `{tool_name}`",
                        state="running",
                    )

                if isinstance(chunk, AIMessage) and chunk.content:
                    yield chunk.content
        finally:
            status_box.update(label="✅ Done", state="complete")

    try:
        ai_message = assistant_container.write_stream(_ai_stream())
    except EnvironmentError as exc:
        ai_message = f"⚠️ Configuration error: {exc}"
        assistant_container.markdown(ai_message)
        status_box.update(label="❌ Error", state="error")
    except Exception as exc:
        ai_message = f"⚠️ Error: {exc}"
        assistant_container.markdown(ai_message)
        status_box.update(label="❌ Error", state="error")

    st.session_state.message_history.append(
        {"role": "assistant", "content": ai_message or ""}
    )
