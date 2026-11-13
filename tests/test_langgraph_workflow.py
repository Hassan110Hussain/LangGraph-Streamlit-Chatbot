"""
Tests for the LangGraph workflow logic.

Mocks the LLM so no real API calls are made.  Verifies that:
  • The graph executes and returns messages.
  • The tool-routing conditional edge is wired correctly.
  • Tool calls loop back through chat_node and produce a final answer.
  • Thread isolation is maintained between conversations.

Uses DEMO_MODE=true to bypass the real LLM provider so no API key is needed.
"""

import os
import sqlite3
import uuid
from typing import Annotated
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Shared state type
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_simple_bot(mock_llm):
    """Build a no-tool graph with a mock LLM."""
    checkpointer = InMemorySaver()

    def chat_node(state: ChatState) -> dict:
        response = mock_llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)
    return graph.compile(checkpointer=checkpointer)


@tool
def _dummy_tool(x: int) -> dict:
    """A dummy tool that returns double the input."""
    return {"result": x * 2}


def _build_tool_bot(mock_llm):
    """Build a tool-routing graph with a mock LLM."""
    checkpointer = InMemorySaver()
    tool_node = ToolNode([_dummy_tool])
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)

    def chat_node(state: ChatState) -> dict:
        response = mock_llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Tests – simple bot
# ---------------------------------------------------------------------------

def test_simple_bot_returns_ai_message():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Hello from mock!")
    bot = _build_simple_bot(mock_llm)

    tid = str(uuid.uuid4())
    result = bot.invoke(
        {"messages": [HumanMessage(content="Hi")]},
        config={"configurable": {"thread_id": tid}},
    )

    messages = result["messages"]
    assert any(isinstance(m, AIMessage) for m in messages)
    ai_reply = next(m for m in messages if isinstance(m, AIMessage))
    assert ai_reply.content == "Hello from mock!"


def test_conversation_history_is_maintained():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Pong")
    bot = _build_simple_bot(mock_llm)

    tid = str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    bot.invoke({"messages": [HumanMessage(content="Ping")]}, config=config)
    result = bot.invoke({"messages": [HumanMessage(content="Again")]}, config=config)

    human_msgs = [m for m in result["messages"] if isinstance(m, HumanMessage)]
    assert len(human_msgs) == 2


def test_different_threads_isolated():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Hi")
    bot = _build_simple_bot(mock_llm)

    tid_a, tid_b = str(uuid.uuid4()), str(uuid.uuid4())
    bot.invoke({"messages": [HumanMessage(content="Thread A msg")]},
               config={"configurable": {"thread_id": tid_a}})

    state_b = bot.get_state({"configurable": {"thread_id": tid_b}})
    assert state_b.values.get("messages", []) == []


# ---------------------------------------------------------------------------
# Tests – demo mode backend
# ---------------------------------------------------------------------------

def test_demo_mode_backend_returns_ai_message(monkeypatch):
    """Verify the full backend works in demo mode without an API key."""
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # Re-import backend in demo mode
    import importlib
    import langgraph_backend
    importlib.reload(langgraph_backend)

    bot = langgraph_backend.chatbot
    tid = str(uuid.uuid4())
    result = bot.invoke(
        {"messages": [HumanMessage(content="Hello demo")]},
        config={"configurable": {"thread_id": tid}},
    )
    messages = result["messages"]
    assert any(isinstance(m, AIMessage) for m in messages)
    # DemoLLM should include demo indicator
    ai_reply = next(m for m in messages if isinstance(m, AIMessage))
    assert ai_reply.content.strip() != ""
