"""
Backend 5 – Async multi-tool chatbot.

Demonstrates the async variant of LangGraph: all LLM and tool calls use
`ainvoke` / `astream`, making it suitable for async web frameworks
(FastAPI, Starlette, etc.).  Runs standalone via `python chatbot_async.py`.

Supports Google Gemini, OpenAI, and Demo Mode via llm_config.
"""

import asyncio
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
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
    Perform a simple arithmetic operation between two numbers.

    Args:
        first_num: The first operand.
        second_num: The second operand.
        operation: One of 'add', 'sub', 'mul', 'div'.

    Returns:
        A dict with result — or 'error'.
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


tools = [calculator]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    """Build and compile the async LangGraph chatbot."""

    async def chat_node(state: ChatState) -> dict:
        """Async LLM node."""
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    _graph = StateGraph(ChatState)
    _graph.add_node("chat_node", chat_node)
    _graph.add_node("tools", tool_node)

    _graph.add_edge(START, "chat_node")
    _graph.add_conditional_edges("chat_node", tools_condition)
    _graph.add_edge("tools", "chat_node")

    return _graph.compile()


# ---------------------------------------------------------------------------
# Demo entry-point
# ---------------------------------------------------------------------------
async def main() -> None:
    """Run a single example turn to verify the graph works end-to-end."""
    bot = build_graph()
    result = await bot.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Find the modulus of 665547557 divided by 23 "
                        "and explain the result like a cricket commentator."
                    )
                )
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
