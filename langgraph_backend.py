"""
Backend 1 – Simple in-memory chatbot (no persistence).

Demonstrates the minimal LangGraph setup: a single LLM node, an in-memory
checkpointer, and no tools.  Every conversation lives only as long as the
Python process — perfect for quick experiments.

Supports Google Gemini, OpenAI, and Demo Mode via llm_config.
"""

from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
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
# Graph
# ---------------------------------------------------------------------------
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
