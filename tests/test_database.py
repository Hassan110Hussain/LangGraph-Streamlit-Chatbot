"""
Tests for the SQLite persistence layer.

Uses an isolated in-memory SQLite database so tests never touch chatbot.db.
Mocks the OpenAI LLM so no API key or network access is needed.

Uses DEMO_MODE=true so backend module imports succeed without credentials.
"""

import sqlite3
import uuid
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Minimal graph fixture (isolated from the module-level singletons)
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def _build_test_chatbot(conn: sqlite3.Connection):
    """Build a minimal LangGraph chatbot backed by an in-memory SQLite DB."""
    checkpointer = SqliteSaver(conn=conn)

    def _mock_chat_node(state: ChatState) -> dict:
        # Return a fixed reply without calling any LLM
        return {"messages": [AIMessage(content="Mock response.")]}

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", _mock_chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)
    return graph.compile(checkpointer=checkpointer), checkpointer, conn


@pytest.fixture
def test_chatbot():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    bot, checkpointer, conn = _build_test_chatbot(conn)
    yield bot, checkpointer, conn
    conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_new_thread_creates_checkpoint(test_chatbot):
    bot, checkpointer, conn = test_chatbot
    tid = str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    bot.invoke({"messages": [HumanMessage(content="Hello")]}, config=config)

    cursor = conn.cursor()
    cursor.execute("SELECT thread_id FROM checkpoints WHERE thread_id = ?", (tid,))
    rows = cursor.fetchall()
    assert len(rows) >= 1


def test_multiple_turns_accumulate_messages(test_chatbot):
    bot, checkpointer, conn = test_chatbot
    tid = str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    bot.invoke({"messages": [HumanMessage(content="Turn 1")]}, config=config)
    bot.invoke({"messages": [HumanMessage(content="Turn 2")]}, config=config)

    state = bot.get_state(config)
    messages = state.values.get("messages", [])
    human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    assert len(human_msgs) == 2


def test_different_threads_are_isolated(test_chatbot):
    bot, checkpointer, conn = test_chatbot
    tid_a = str(uuid.uuid4())
    tid_b = str(uuid.uuid4())

    bot.invoke({"messages": [HumanMessage(content="Thread A")]},
               config={"configurable": {"thread_id": tid_a}})
    bot.invoke({"messages": [HumanMessage(content="Thread B")]},
               config={"configurable": {"thread_id": tid_b}})

    state_a = bot.get_state({"configurable": {"thread_id": tid_a}})
    state_b = bot.get_state({"configurable": {"thread_id": tid_b}})

    msgs_a = [m.content for m in state_a.values.get("messages", []) if isinstance(m, HumanMessage)]
    msgs_b = [m.content for m in state_b.values.get("messages", []) if isinstance(m, HumanMessage)]

    assert "Thread A" in msgs_a
    assert "Thread B" not in msgs_a
    assert "Thread B" in msgs_b
    assert "Thread A" not in msgs_b


def test_retrieve_all_threads(test_chatbot):
    bot, checkpointer, conn = test_chatbot
    tids = [str(uuid.uuid4()) for _ in range(3)]

    for tid in tids:
        bot.invoke({"messages": [HumanMessage(content="hi")]},
                   config={"configurable": {"thread_id": tid}})

    cursor = conn.cursor()
    cursor.execute(
        "SELECT thread_id, MAX(rowid) FROM checkpoints GROUP BY thread_id ORDER BY MAX(rowid) DESC"
    )
    rows = cursor.fetchall()
    retrieved = [r[0] for r in rows]

    for tid in tids:
        assert tid in retrieved


def test_empty_thread_returns_no_messages(test_chatbot):
    bot, checkpointer, conn = test_chatbot
    tid = str(uuid.uuid4())
    state = bot.get_state({"configurable": {"thread_id": tid}})
    assert state.values.get("messages", []) == []


def test_messages_survive_reconnect():
    """Simulate an app restart by closing and reopening the SQLite connection."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        tid = str(uuid.uuid4())
        config = {"configurable": {"thread_id": tid}}

        # First session
        conn1 = sqlite3.connect(db_path, check_same_thread=False)
        bot1, _, _ = _build_test_chatbot(conn1)
        bot1.invoke({"messages": [HumanMessage(content="Persist me")]}, config=config)
        conn1.close()

        # Second session — new connection, same DB file
        conn2 = sqlite3.connect(db_path, check_same_thread=False)
        bot2, _, _ = _build_test_chatbot(conn2)
        state = bot2.get_state(config)
        messages = state.values.get("messages", [])
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        assert any("Persist me" in m.content for m in human_msgs)
        conn2.close()
    finally:
        os.unlink(db_path)
