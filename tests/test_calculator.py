"""
Tests for the calculator tool shared across all tool-enabled backends.

We import directly from chatbot.py (Backend 3) which is the canonical
definition.  The same logic lives in langgraph_backend_tool.py and
chatbot_async.py — a future refactor could extract it to a shared module.
"""

import pytest

# Import the underlying function.  The @tool decorator wraps it but the
# original function is accessible via .func on the LangChain tool object.
from chatbot import calculator as _calculator_tool

# Unwrap the LangChain StructuredTool to get the plain Python function.
calc = _calculator_tool.func


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def test_add():
    result = calc(first_num=10, second_num=5, operation="add")
    assert result["result"] == 15
    assert result["operation"] == "add"


def test_subtract():
    result = calc(first_num=10, second_num=3, operation="sub")
    assert result["result"] == 7


def test_multiply():
    result = calc(first_num=4, second_num=7, operation="mul")
    assert result["result"] == 28


def test_divide():
    result = calc(first_num=20, second_num=4, operation="div")
    assert result["result"] == 5.0


def test_divide_float():
    result = calc(first_num=1, second_num=3, operation="div")
    assert abs(result["result"] - 0.3333) < 0.0001


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------

def test_division_by_zero():
    result = calc(first_num=10, second_num=0, operation="div")
    assert "error" in result
    assert "zero" in result["error"].lower()


def test_unsupported_operation():
    result = calc(first_num=10, second_num=5, operation="mod")
    assert "error" in result
    assert "mod" in result["error"]


def test_negative_numbers():
    result = calc(first_num=-5, second_num=3, operation="add")
    assert result["result"] == -2


def test_zero_operand_multiply():
    result = calc(first_num=0, second_num=999, operation="mul")
    assert result["result"] == 0
