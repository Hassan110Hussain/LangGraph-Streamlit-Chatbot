# 🤖 LangGraph Chatbot UI

A modular, progressively complex AI chatbot built with **LangGraph**, **LangChain**, and **Streamlit**.

Supports **Google Gemini** and **OpenAI** as AI providers, plus a **Demo Mode** that lets anyone explore the app without any API key.

> **Designed for learning and showcasing** — ideal for GitHub portfolios, technical interviews, and exploring LangGraph architecture.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🤖 Dual AI providers | Google Gemini 2.0 Flash and OpenAI GPT-4o-mini |
| 🎭 Demo mode | Full UI exploration with no API key required |
| 5 backend implementations | From minimal in-memory to full async multi-tool |
| 4 Streamlit frontends | From basic to streaming + tools + live status |
| 🔧 Tool calling | Calculator, DuckDuckGo search, Alpha Vantage stocks |
| 💾 SQLite persistence | Conversations survive process restarts |
| 🗂️ Thread management | Create, switch, and restore named conversations |
| 🌊 Token streaming | Real-time response streaming with st.write_stream |
| 🔒 Secure credential handling | Keys only from environment variables, never hardcoded |
| 🧪 48 passing tests | No API key needed to run the full test suite |

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| LLM orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) · [LangChain](https://github.com/langchain-ai/langchain) |
| AI providers | [Google Gemini](https://ai.google.dev/) via `langchain-google-genai` · [OpenAI](https://platform.openai.com/) via `langchain-openai` |
| Persistence | SQLite via `langgraph-checkpoint-sqlite` |
| Frontend | [Streamlit](https://streamlit.io/) |
| Web search | DuckDuckGo via `langchain-community` |
| Stock prices | [Alpha Vantage API](https://www.alphavantage.co/) |
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
                          │     llm_config      │  ◄── Provider selection
                          │  (Gemini/OpenAI/    │       (env var or auto-detect)
                          │   Demo)             │
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │     chat_node       │  ◄── LLM call
                          └─────────┬──────────┘
                                    │ tools_condition
                          ┌─────────▼──────────┐
                          │     tool_node       │  ◄── calculator / search / stocks
                          └─────────┬──────────┘
                                    │ (loops back until done)
                          ┌─────────▼──────────┐
                          │  SqliteSaver /      │
                          │  InMemorySaver      │
                          └────────────────────┘
```

Each LangGraph checkpoint stores the full message history keyed by **thread ID** (UUID4), enabling multi-conversation management with SQLite persistence.

---

## 📂 Project Structure

```
chatbot_ui/
│
├── llm_config.py                    # Centralized LLM provider config (NEW)
│
├── backends/
│   ├── langgraph_backend.py          # Backend 1 – in-memory, no tools
│   ├── langgraph_backend_database.py # Backend 2 – SQLite, no tools
│   ├── chatbot.py                    # Backend 3 – calculator + SQLite
│   ├── langgraph_backend_tool.py     # Backend 4 – multi-tool + SQLite
│   └── chatbot_async.py              # Backend 5 – async variant
│
├── frontends/
│   ├── streamlit_frontend.py             # Frontend 1 – minimal
│   ├── streamlit_frontend_streaming.py   # Frontend 2 – streaming
│   ├── streamlit_frontend_database.py    # Frontend 3 – persistent conversations
│   └── streamlit_frontend_threading.py   # Frontend 4 – tools + live status
│
├── tests/
│   ├── conftest.py              # Sets DEMO_MODE=true for all tests
│   ├── test_calculator.py       # Calculator tool tests
│   ├── test_database.py         # SQLite persistence tests
│   ├── test_langgraph_workflow.py # Graph execution + demo mode tests
│   ├── test_llm_config.py       # Provider config + demo mode tests (NEW)
│   └── test_stock_tool.py       # Stock tool tests (mocked HTTP)
│
├── .env.example                 # Safe template — commit this
├── .gitignore                   # Excludes .env, *.db, .venv/
├── requirements.txt             # Pinned dependencies
└── README.md
```

> Note: all Python files are in the project root. The structure above groups them conceptually.

---

## ⚙️ Installation

### 1. Requirements

- Python **3.11–3.14** (tested on 3.14.4, Windows x64)
- pip

> Python 3.14 compatibility is confirmed for all packages in `requirements.txt`. The `pydantic-core` Rust compilation issue that affected earlier setups is resolved by the pinned dependency versions.

### 2. Clone the repository

```bash
git clone https://github.com/your-username/chatbot_ui.git
cd chatbot_ui
```

### 3. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Then edit `.env` and fill in your credentials (see the sections below).

---

## 🎭 Running in Demo Mode (No API Key Required)

Demo mode lets you explore the full application — conversation history, thread switching, SQLite persistence, the tool UI, and streaming — without any API key.

**Edit `.env`:**
```env
DEMO_MODE=true
```

Then launch any frontend:
```bash
streamlit run streamlit_frontend_threading.py
```

Responses are clearly labelled **[DEMO MODE — Simulated Response]** and are never presented as real AI output.

---

## 🔑 API Key Setup

### Google Gemini (recommended)

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and create a free API key.
2. Edit `.env`:
   ```env
   AI_PROVIDER=gemini
   GOOGLE_API_KEY=your_actual_key_here
   DEMO_MODE=false
   ```

The default model is `gemini-2.0-flash`. Override with `GEMINI_MODEL=gemini-1.5-pro` if needed.

### OpenAI

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys) and create a key.
2. Edit `.env`:
   ```env
   AI_PROVIDER=openai
   OPENAI_API_KEY=your_actual_key_here
   DEMO_MODE=false
   ```

The default model is `gpt-4o-mini`. Override with `OPENAI_MODEL=gpt-4o` if needed.

### Selecting a provider

The `AI_PROVIDER` variable controls which provider is used. If it is not set, the app auto-detects:
- Sets `gemini` if `GOOGLE_API_KEY` is present
- Sets `openai` if `OPENAI_API_KEY` is present

**You only need ONE provider's key** — the other can remain blank.

### Alpha Vantage (optional — for stock price tool)

1. Get a free key at [alphavantage.co](https://www.alphavantage.co/support/#api-key).
2. Edit `.env`:
   ```env
   ALPHA_VANTAGE_API_KEY=your_actual_key_here
   ```

The stock tool shows a clear error message if this key is missing.

---

## 🚀 Running the Application

Each frontend is standalone — pick the complexity level you want to explore.

### Frontend 1 — Minimal (single thread, no persistence)
```bash
streamlit run streamlit_frontend.py
```
Pairs with `langgraph_backend.py`.

### Frontend 2 — Streaming
```bash
streamlit run streamlit_frontend_streaming.py
```
Pairs with `langgraph_backend.py`. Token-by-token streaming.

### Frontend 3 — Persistent Conversations
```bash
streamlit run streamlit_frontend_database.py
```
Pairs with `langgraph_backend_database.py`. Conversations survive restarts.

### Frontend 4 — Full-Featured (Tools + Live Status) ⭐ Recommended
```bash
streamlit run streamlit_frontend_threading.py
```
Pairs with `langgraph_backend_tool.py`. Web search, stock prices, calculator, live tool status indicator.

### Backend 5 — Async (terminal only)
```bash
python chatbot_async.py
```
Demonstrates `ainvoke` / `astream` for async frameworks.

---

## 🔧 Available Tools (Backends 3, 4, 5)

| Tool | Requires key | Description |
|---|---|---|
| `calculator` | No | Add, subtract, multiply, divide two numbers |
| `duckduckgo_search` | No | Real-time web search via DuckDuckGo |
| `get_stock_price` | `ALPHA_VANTAGE_API_KEY` | Latest price for any ticker (AAPL, TSLA…) |

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

## 💾 SQLite Persistence

Conversations are stored in `chatbot.db` (SQLite). LangGraph's `SqliteSaver` checkpointer serialises the full message state keyed by a UUID4 thread ID.

- Each new conversation gets a fresh UUID thread ID
- Messages accumulate within a thread across multiple turns
- Switching threads reloads the exact conversation from the database
- Conversations survive app restarts
- Tool messages are filtered from the display layer (not shown as AI responses)
- `chatbot.db` is excluded from git via `.gitignore`

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

No API key required — the test suite runs entirely in demo mode.

```
tests/test_calculator.py         ✓  9 tests  — calculator arithmetic
tests/test_database.py           ✓  6 tests  — SQLite persistence
tests/test_langgraph_workflow.py ✓  4 tests  — graph execution
tests/test_llm_config.py         ✓ 23 tests  — provider config & demo mode
tests/test_stock_tool.py         ✓  6 tests  — stock tool (mocked HTTP)
─────────────────────────────────────────────
Total                               48 tests  — all passing
```

---

## 🔒 Security Notes

- **Never commit `.env`** — it is excluded by `.gitignore`
- **`.env.example` is safe to commit** — it contains no real credentials
- API keys are read exclusively from environment variables
- Keys are never printed in logs or displayed in the UI
- The stock tool uses `ALPHA_VANTAGE_API_KEY` from the environment only

If you accidentally commit a key, revoke it immediately at the provider's dashboard and generate a new one.

---

## 🐛 Troubleshooting

### `EnvironmentError: GOOGLE_API_KEY is not set`
Add your Gemini key to `.env` or set `DEMO_MODE=true` to bypass.

### `EnvironmentError: OPENAI_API_KEY is not set`
Add your OpenAI key to `.env` or switch to `AI_PROVIDER=gemini`.

### `ModuleNotFoundError: No module named 'langgraph.checkpoint.sqlite'`
Run: `pip install langgraph-checkpoint-sqlite==2.0.11`

### `ModuleNotFoundError: No module named 'langchain_google_genai'`
Run: `pip install langchain-google-genai==2.0.11`

### Streamlit shows a blank page or crashes on import
Ensure you activated the virtual environment before running Streamlit:
```powershell
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
```

### `protobuf` version conflicts
The pinned versions in `requirements.txt` resolve all known protobuf conflicts between `streamlit==1.44.1` and `langchain-google-genai==2.0.11`. Run:
```bash
pip install -r requirements.txt --force-reinstall
```

### DuckDuckGo search fails intermittently
DuckDuckGo rate-limits are enforced at the network level. Wait a few seconds and retry. No API key is required for this tool.

---

## 📸 Screenshots

_Add screenshots of each frontend here after running the app._

---

## 🔮 Future Improvements

- [ ] System prompt configuration via UI
- [ ] Export conversation to Markdown / PDF
- [ ] Anthropic Claude provider support
- [ ] FastAPI backend for production serving
- [ ] Docker + docker-compose setup
- [ ] Conversation search
- [ ] Message editing and regeneration

---

## 📄 License

MIT
