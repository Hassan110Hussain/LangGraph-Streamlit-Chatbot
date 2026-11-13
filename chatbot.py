"""
Backend 3 – Tool-integrated chatbot with SQLite persistence.

Builds on Backend 2 by adding a calculator tool and conditional routing:
the LLM can decide to call the tool, whose result is fed back for a final
answer.  Uses the same SQLite database as the other backends.

Supports Google Gemini, OpenAI, and Demo Mode via llm_config.
Note: In Demo Mode the calculator still executes normally; only LLM calls
are simulated.
"""

import sqlite3
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from llm_config import get_llm

load_dotenv()

# ---------------------------------------------------------------------------
# Model — provider resolved from environment (Gemini / OpenAI / Demo)
# ---------------------------------------------------------------------------
llm = get_llm()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.

    Args:
        first_num: The first operand.
        second_num: The second operand.
        operation: One of 'add', 'sub', 'mul', 'div'.

    Returns:
        A dict with keys first_num, second_num, operation, result — or 'error'.
    """
    ops = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b if b != 0 else None,
    }
    if operation not in ops:
        return {"error": f"Unsupported operation '{operation}'. Use: add, sub, mul, div."}
    if operation == "div" and second_num == 0:
        return {"error": "Division by zero is not allowed."}

    result = ops[operation](first_num, second_num)
    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": result,
    }


tools = [calculator]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def chat_node(state: ChatState) -> dict:
    """LLM node — may answer directly or request a tool call."""
    response = llm_with_tools.invoke(state["messages"])
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
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

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
