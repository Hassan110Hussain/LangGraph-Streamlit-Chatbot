# 🤖 LangGraph Chatbot UI

A modular, progressively complex chatbot built with **LangGraph**, **LangChain**, and **Streamlit**.  
Each backend demonstrates a different architectural pattern — from the simplest in-memory LLM call to a fully persistent, multi-tool agent with live streaming.

> **Designed to be educational and showcase-ready** — ideal for GitHub portfolios, LinkedIn posts, and technical interviews.

---

## ✨ Key Features

- 5 backend implementations with progressive complexity
- 4 Streamlit frontends with increasing UX sophistication
- LangGraph state machine with conditional tool routing
- SQLite-based conversation persistence across process restarts
- Token-by-token streaming with live tool status indicators
- Web search, stock price lookup, and calculator tools
- Clean environment variable configuration (no hardcoded secrets)
- Full pytest test suite (no API key required)

---

## 🛠️ Technologies

| Layer | Library |
|---|---|
| LLM orchestration | [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain) |
| Language model | OpenAI (`gpt-4o-mini` by default) |
| Web search tool | DuckDuckGo via `langchain-community` |
| Stock price tool | [Alpha Vantage API](https://www.alphavantage.co/) |
| Persistence | SQLite + `langgraph.checkpoint.sqlite.SqliteSaver` |
| Frontend | [Streamlit](https://streamlit.io/) |
| Testing | pytest |

---

## 🏗️ Architecture

```
User input
    │
    ▼
Streamlit frontend  ──────►  LangGraph chatbot (backend)
                                    │
                          ┌─────────▼──────────┐
                          │     chat_node       │  ◄── OpenAI LLM
                          └─────────┬──────────┘
                                    │ tools_condition
                          ┌─────────▼──────────┐
                          │     tool_node       │  ◄── calculator / search / stocks
                          └─────────┬──────────┘
                                    │ (loops back)
                          ┌─────────▼──────────┐
                          │   SqliteSaver /     │
                          │   InMemorySaver     │
                          └────────────────────┘
```

Each LangGraph checkpoint stores the full message history keyed by **thread ID** (UUID4), enabling multi-conversation management.

---

## 📂 Project Structure

```
chatbot_ui/
│
├── backends/
│   ├── langgraph_backend.py          # Backend 1 – in-memory, no tools
│   ├── langgraph_backend_database.py # Backend 2 – SQLite persistence, no tools
│   ├── chatbot.py                    # Backend 3 – calculator tool + SQLite
│   ├── langgraph_backend_tool.py     # Backend 4 – multi-tool (search, stocks, calc)
│   └── chatbot_async.py              # Backend 5 – async/await variant
│
├── frontends/
│   ├── streamlit_frontend.py         # Frontend 1 – minimal, single thread
│   ├── streamlit_frontend_streaming.py  # Frontend 2 – token streaming
│   ├── streamlit_frontend_database.py   # Frontend 3 – persistent conversations
│   └── streamlit_frontend_threading.py # Frontend 4 – tools + live status
│
├── tests/
│   ├── test_calculator.py            # Calculator tool unit tests
│   ├── test_stock_tool.py            # Stock tool tests (mocked HTTP)
│   ├── test_database.py              # SQLite persistence tests
│   └── test_langgraph_workflow.py    # LangGraph graph execution tests
│
├── .env.example                      # Required environment variables template
├── .gitignore
├── requirements.txt
└── README.md
```

> Note: all Python files are in the project root (not subdirectories) — the structure above is conceptual grouping for clarity.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/chatbot_ui.git
cd chatbot_ui
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | Your OpenAI API key |
| `ALPHA_VANTAGE_API_KEY` | ⚠️ Optional | Needed for stock price lookups only |

Get an OpenAI key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).  
Get a free Alpha Vantage key at [alphavantage.co](https://www.alphavantage.co/support/#api-key).

---

## 🚀 Running the Application

Each frontend is standalone — pick the one that matches the complexity level you want to explore.

### Frontend 1 — Minimal (no persistence, no tools)
```bash
streamlit run streamlit_frontend.py
```
Pairs with `langgraph_backend.py`

---

### Frontend 2 — Streaming
```bash
streamlit run streamlit_frontend_streaming.py
```
Pairs with `langgraph_backend.py` · demonstrates token-by-token streaming.

---

### Frontend 3 — Persistent Conversations
```bash
streamlit run streamlit_frontend_database.py
```
Pairs with `langgraph_backend_database.py` · conversations survive restarts.

---

### Frontend 4 — Full-Featured (Tools + Live Status)
```bash
streamlit run streamlit_frontend_threading.py
```
Pairs with `langgraph_backend_tool.py` · web search, stock prices, calculator.

---

### Async backend demo (terminal only)
```bash
python chatbot_async.py
```

---

## 🔧 Available Tools (Backend 4)

| Tool | Description |
|---|---|
| `calculator` | Add, subtract, multiply, divide two numbers |
| `duckduckgo_search` | Search the web in real time |
| `get_stock_price` | Fetch the latest price for any ticker via Alpha Vantage |

---

## 💾 Conversation Persistence

Conversations are stored in `chatbot.db` (SQLite).  LangGraph's `SqliteSaver` checkpointer serialises the full message state keyed by a UUID4 thread ID.

- Each new conversation gets a fresh UUID thread ID.
- Messages accumulate within a thread across multiple turns.
- Switching threads reloads the exact conversation from the database.
- The file is excluded from git via `.gitignore`.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

No API key required — all external calls are mocked.

```
tests/test_calculator.py        ✓  9 tests
tests/test_stock_tool.py        ✓  5 tests
tests/test_database.py          ✓  5 tests
tests/test_langgraph_workflow.py ✓  3 tests
```

---

## 🔄 Backend Comparison

| | Backend 1 | Backend 2 | Backend 3 | Backend 4 | Backend 5 |
|---|---|---|---|---|---|
| File | `langgraph_backend.py` | `langgraph_backend_database.py` | `chatbot.py` | `langgraph_backend_tool.py` | `chatbot_async.py` |
| Persistence | In-memory | SQLite | SQLite | SQLite | None |
| Tools | None | None | Calculator | Calc + Search + Stocks | Calculator |
| Async | No | No | No | No | Yes |
| Streaming | Yes | Yes | Yes | Yes | Via `astream` |

---

## 📸 Screenshots

_Add screenshots of each frontend here after running the app._

---

## 🔮 Future Improvements

- [ ] System prompt configuration via UI
- [ ] Export conversation to Markdown / PDF
- [ ] Support for additional LLM providers (Anthropic, Ollama)
- [ ] FastAPI backend for production serving
- [ ] Docker + docker-compose setup
- [ ] Conversation search
- [ ] Message editing and regeneration

---

## 📄 License

MIT
