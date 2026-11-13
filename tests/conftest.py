"""
Pytest configuration for the LangGraph Chatbot UI test suite.

Sets DEMO_MODE=true before any module is imported, so all backend
modules initialise with the DemoLLM stub instead of calling a real
AI provider.  This means no API key is required to run the tests.
"""

import os

# Must be set before any backend module is imported — pytest loads conftest.py
# first, before collecting test modules.
os.environ.setdefault("DEMO_MODE", "true")

# Ensure neither real provider key is accidentally active
# (only clear if they aren't already set by the user deliberately)
if os.environ.get("DEMO_MODE", "").lower() in ("true", "1", "yes"):
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("GOOGLE_API_KEY", None)
