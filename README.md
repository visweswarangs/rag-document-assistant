[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-document-assistant-fxwshncymsf3aqt8g4atmf.streamlit.app/)

# 📚 RAG Document Assistant

A production-ready, self-contained **Retrieval-Augmented Generation (RAG)** application that lets you chat with your own PDF and text documents — powered by Google Gemini, ChromaDB, and Streamlit.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Document ingestion** | Upload `.pdf` and `.txt` files via the sidebar |
| **Smart chunking** | Overlapping 800-char chunks with metadata tracking (filename, page, chunk index) |
| **Semantic embeddings** | `text-embedding-004` via the official Google GenAI SDK |
| **In-memory vector store** | Ephemeral ChromaDB collection — no disk setup needed |
| **Grounded chat** | `gemini-2.5-flash` answers strictly from retrieved context |
| **Source citations** | Expandable panel shows exact chunks + similarity scores used |
| **Multi-document support** | Embed multiple files into the same session |
| **Flexible key management** | `st.secrets` → `.env` → sidebar manual input |

---

## 🗂️ Project Structure

```
RAG/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .env                 # Your real API key (gitignored – create from .env.example)
├── .gitignore           # Excludes .env, __pycache__, etc.
└── README.md            # This file
```

---

## 🚀 Quick Start

### 1 · Clone / Copy the project

```bash
# If using git
git clone <your-repo-url> RAG
cd RAG

# Or just navigate to the folder
cd C:\Projects\RAG
```

### 2 · Create a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

### 4 · Set your API key

**Option A — `.env` file (recommended for local dev)**

```bash
# Copy the template
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux

# Open .env and replace the placeholder with your real key
GEMINI_API_KEY=AIza...your_key_here...
```

**Option B — Streamlit secrets (recommended for deployment)**

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "AIza...your_key_here..."
```

**Option C — Sidebar input**

Leave `.env` empty and paste your key directly in the app sidebar. Useful for quick demos.

> 🔑 Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey).

### 5 · Run the app

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 🧠 Architecture

```
 User ──► [Chat Input]
              │
              ▼
    ┌─────────────────────┐
    │  Embed Question     │  text-embedding-004
    │  (Google GenAI SDK) │
    └────────┬────────────┘
             │  query vector
             ▼
    ┌─────────────────────┐
    │  ChromaDB           │  cosine similarity search
    │  (in-memory)        │  → top-4 chunks
    └────────┬────────────┘
             │  context chunks
             ▼
    ┌─────────────────────┐
    │  Prompt builder     │  system prompt + chunks + question
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │  gemini-2.5-flash   │  grounded answer generation
    └────────┬────────────┘
             │
             ▼
       [Answer + Citations]
```

### Key parameters (edit in `app.py`)

| Constant | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `4` | Retrieved chunks per query |
| `EMBEDDING_MODEL` | `text-embedding-004` | Gemini embedding model |
| `CHAT_MODEL` | `gemini-2.5-flash` | Gemini chat model |

---

## 📋 Usage Walkthrough

1. **Open the app** → `http://localhost:8501`
2. **Enter your API key** (if not in `.env` / secrets)
3. **Upload documents** in the left sidebar (PDF or TXT)
4. **Click "⚙️ Process & Embed Documents"** — a progress bar shows chunking & embedding progress
5. **Ask questions** in the chat box at the bottom
6. **Expand "📎 View Source Citations"** under any answer to see the exact text chunks used

---

## 🔒 Security Notes

- Your API key is **never stored on disk** by the app itself
- The `.env` file is intentionally excluded from git via `.gitignore`
- ChromaDB runs **fully in-memory** — no data is written to disk
- All document content stays local; only embedding and generation requests are sent to Google's API

---

## 🛠️ Troubleshooting

| Issue | Fix |
|---|---|
| `No API key found` | Check `.env` has `GEMINI_API_KEY=...` (no quotes around value) |
| PDF uploads fail | Ensure `pypdf` is installed: `pip install pypdf` |
| Empty answers | Try rephrasing your question; the answer must be in the document |
| ChromaDB errors | Restart the app — the in-memory store resets cleanly |
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` in your virtual environment |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `google-genai` | Official Google GenAI SDK (embeddings + chat) |
| `chromadb` | In-memory vector database |
| `pypdf` | PDF text extraction |
| `python-dotenv` | Load `.env` files |

---

## 📄 License

MIT — free to use, modify, and distribute.
