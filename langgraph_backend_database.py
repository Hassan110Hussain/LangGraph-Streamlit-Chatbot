"""
Backend 2 – SQLite-persisted chatbot (no tools).

Demonstrates how to swap InMemorySaver for SqliteSaver so conversations
survive process restarts.  Thread IDs are UUIDs managed by the frontend.

Supports Google Gemini, OpenAI, and Demo Mode via llm_config.
"""

import sqlite3
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from llm_config import get_llm

load_dotenv()

# ---------------------------------------------------------------------------
# Model — provider resolved from environment (Gemini / OpenAI / Demo)
# ---------------------------------------------------------------------------
llm = get_llm()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def chat_node(state: ChatState) -> dict:
    """Invoke the LLM with the current message history and return the reply."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def retrieve_all_threads() -> list[str]:
    """Return thread IDs ordered by most-recently updated first."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT thread_id, MAX(rowid) AS last_rowid
        FROM checkpoints
        GROUP BY thread_id
        ORDER BY last_rowid DESC
        """
    )
    rows = cursor.fetchall()
    return [row[0] for row in rows]
