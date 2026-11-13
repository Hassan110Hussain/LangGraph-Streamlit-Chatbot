"""
Backend 4 – Multi-tool chatbot with SQLite persistence.

Extends Backend 3 with two additional real-world tools:
  • DuckDuckGo web search
  • Alpha Vantage stock price lookup

The API key for Alpha Vantage is loaded from the environment variable
ALPHA_VANTAGE_API_KEY — never hard-coded.

Supports Google Gemini, OpenAI, and Demo Mode via llm_config.
Note: In Demo Mode the tools still execute; only LLM calls are simulated.
"""

import os
import sqlite3
from typing import Annotated

import requests
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
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
search_tool = DuckDuckGoSearchRun(region="us-en")


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

    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": ops[operation](first_num, second_num),
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given ticker symbol (e.g. 'AAPL', 'TSLA').

    Requires the ALPHA_VANTAGE_API_KEY environment variable.

    Args:
        symbol: The stock ticker symbol.

    Returns:
        A dict with the Global Quote data — or 'error' on failure.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    if not api_key:
        return {
            "error": (
                "ALPHA_VANTAGE_API_KEY environment variable is not set. "
                "Get a free key at https://www.alphavantage.co/support/#api-key"
            )
        }

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Alpha Vantage returns a "Note" key when the rate limit is hit
        if "Note" in data:
            return {
                "error": "Alpha Vantage rate limit reached. Please wait a minute and try again."
            }
        if "Information" in data:
            return {"error": f"Alpha Vantage API info: {data['Information']}"}
        if "Global Quote" not in data or not data["Global Quote"]:
            return {"error": f"No data found for symbol '{symbol}'. Check the ticker and try again."}

        return data

    except requests.exceptions.Timeout:
        return {"error": "Stock price request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"error": "Network error: unable to reach Alpha Vantage. Check your connection."}
    except requests.exceptions.RequestException as exc:
        return {"error": f"Stock price request failed: {exc}"}
    except ValueError:
        return {"error": "Received an unexpected response from Alpha Vantage."}


tools = [search_tool, get_stock_price, calculator]
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
    """LLM node — may answer directly or request one or more tool calls."""
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
