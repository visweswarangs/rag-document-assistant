"""
RAG Document Assistant
======================
A production-ready Retrieval-Augmented Generation app built with:
  - Streamlit      (UI)
  - numpy          (pure-Python in-memory vector store — no native extensions)
  - Google GenAI SDK  (embeddings via gemini-embedding-001, chat via gemini-2.5-flash)
"""

import os
import re
import math
import time
import logging
import streamlit as st

from io import BytesIO
from typing import Optional

# ── dotenv (optional, only needed when running outside Streamlit Cloud) ──────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed – rely on env vars or st.secrets

# ── Google GenAI ─────────────────────────────────────────────────────────────
from google import genai

# ── PDF parsing ──────────────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
# Run `python test_embed.py` to confirm which embedding model is available on
# your API key.  Confirmed working models: gemini-embedding-001,
# gemini-embedding-2, gemini-embedding-2-preview
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.8-flash"
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between consecutive chunks
TOP_K = 4               # number of chunks to retrieve per query

# Retry / back-off settings for transient 503 / 429 / 500 API errors
RETRY_MAX_ATTEMPTS = 5
RETRY_BASE_DELAY   = 2.0   # seconds — doubles each attempt (2 → 4 → 8 → 16 …)
RETRY_MAX_DELAY    = 30.0  # seconds — cap per sleep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python vector store (replaces ChromaDB — zero native dependencies)
# ─────────────────────────────────────────────────────────────────────────────
class VectorStore:
    """
    Minimal in-memory cosine-similarity vector store.
    Stores embeddings as plain Python lists — no numpy, no Rust, no C++.
    """

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.embeddings: list[list[float]] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _norm(v: list[float]) -> float:
        return math.sqrt(sum(x * x for x in v))

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        na, nb = self._norm(a), self._norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return self._dot(a, b) / (na * nb)

    # ── public API ────────────────────────────────────────────────────────────
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self.ids.extend(ids)
        self.embeddings.extend(embeddings)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

    def query(
        self, query_embedding: list[float], n_results: int = TOP_K
    ) -> list[dict]:
        """Return the top-n most similar chunks as {text, metadata, distance}."""
        if not self.embeddings:
            return []

        scored = [
            (self._cosine_similarity(query_embedding, emb), doc, meta)
            for emb, doc, meta in zip(self.embeddings, self.documents, self.metadatas)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)  # highest similarity first

        results = []
        for sim, doc, meta in scored[:n_results]:
            results.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "distance": 1.0 - sim,   # distance = 1 - similarity (cosine)
                }
            )
        return results

    def count(self) -> int:
        return len(self.ids)

    def reset(self) -> None:
        self.__init__()


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state() -> None:
    defaults = {
        "chat_history": [],          # list of {"role": str, "content": str}
        "vector_store": None,        # VectorStore instance
        "uploaded_filenames": [],    # list of processed file names
        "total_chunks": 0,           # cumulative chunk count
        "genai_client": None,        # google.genai.Client instance
        "api_key_source": None,      # where the key came from
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()

# ─────────────────────────────────────────────────────────────────────────────
# API Key resolution
# ─────────────────────────────────────────────────────────────────────────────
def resolve_api_key(manual_key: Optional[str] = None) -> Optional[str]:
    """Return the first available Gemini API key and record its source."""
    if manual_key and manual_key.strip():
        st.session_state.api_key_source = "sidebar input"
        return manual_key.strip()
    try:
        key = st.secrets.get("GEMINI_API_KEY", None)
        if key:
            st.session_state.api_key_source = "Streamlit secrets"
            return key
    except Exception:
        pass
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        st.session_state.api_key_source = ".env / environment variable"
        return key
    st.session_state.api_key_source = None
    return None


def get_genai_client(manual_key: Optional[str] = None) -> Optional[genai.Client]:
    """Initialise (or reuse) the Google GenAI client."""
    if manual_key and manual_key.strip():
        api_key = resolve_api_key(manual_key)
        st.session_state.genai_client = genai.Client(api_key=api_key)
        return st.session_state.genai_client
    if st.session_state.genai_client is not None:
        return st.session_state.genai_client
    api_key = resolve_api_key()
    if api_key:
        st.session_state.genai_client = genai.Client(api_key=api_key)
        return st.session_state.genai_client
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Vector store helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_vector_store() -> VectorStore:
    if st.session_state.vector_store is None:
        st.session_state.vector_store = VectorStore()
    return st.session_state.vector_store


def reset_store() -> None:
    """Wipe all documents and reset chat."""
    if st.session_state.vector_store is not None:
        st.session_state.vector_store.reset()
    st.session_state.uploaded_filenames = []
    st.session_state.total_chunks = 0
    st.session_state.chat_history = []


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction & chunking
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> list[dict]:
    """Extract text page-by-page from a PDF. Returns list of {text, page}."""
    if not PYPDF_AVAILABLE:
        st.error("pypdf is not installed. Run `pip install pypdf`.")
        return []
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"text": text, "page": page_num})
    return pages


def extract_text_from_txt(file_bytes: bytes) -> list[dict]:
    """Decode a plain-text file as a single 'page'."""
    text = file_bytes.decode("utf-8", errors="replace").strip()
    return [{"text": text, "page": 1}] if text else []


def split_into_chunks(
    pages: list[dict],
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Sliding-window character-level chunker with overlap.
    Returns chunks with keys: id, text, filename, page, chunk_index.
    """
    chunks = []
    chunk_index = 0

    for page_data in pages:
        text = re.sub(r"\s+", " ", page_data["text"])
        page_num = page_data["page"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "id": f"{filename}_p{page_num}_c{chunk_index}",
                        "text": chunk_text,
                        "filename": filename,
                        "page": page_num,
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1

            if end == len(text):
                break
            start += chunk_size - overlap

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Retry helper (exponential back-off, stdlib only)
# ─────────────────────────────────────────────────────────────────────────────
_RETRYABLE_CODES = {429, 500, 503}   # rate-limit / overload / internal error


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception is a transient API error worth retrying."""
    msg = str(exc)
    return any(str(code) in msg for code in _RETRYABLE_CODES) or \
           any(kw in msg.upper() for kw in ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "INTERNAL"))


def call_with_retry(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) up to RETRY_MAX_ATTEMPTS times.
    Uses exponential back-off on transient API errors (503, 429, 500).
    Raises the last exception if all attempts fail.
    """
    delay = RETRY_BASE_DELAY
    last_exc: Exception | None = None

    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_retryable(exc) or attempt == RETRY_MAX_ATTEMPTS:
                raise
            last_exc = exc
            wait = min(delay, RETRY_MAX_DELAY)
            logger.warning(
                "Transient API error (attempt %d/%d): %s — retrying in %.1fs…",
                attempt, RETRY_MAX_ATTEMPTS, exc, wait,
            )
            time.sleep(wait)
            delay *= 2   # exponential back-off

    raise last_exc  # unreachable but satisfies type checkers


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────
def embed_texts(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of text strings (with automatic retry)."""
    def _call():
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
        )
        return [emb.values for emb in response.embeddings]

    return call_with_retry(_call)


# ─────────────────────────────────────────────────────────────────────────────
# Document ingestion pipeline
# ─────────────────────────────────────────────────────────────────────────────
def ingest_document(uploaded_file, genai_client: genai.Client) -> int:
    """
    Full pipeline: extract → chunk → embed → store.
    Returns the number of chunks added, or -1 on failure.
    """
    filename = uploaded_file.name
    file_bytes = uploaded_file.read()

    # ── Extract ────────────────────────────────────────────────────────────
    if filename.lower().endswith(".pdf"):
        pages = extract_text_from_pdf(file_bytes)
    elif filename.lower().endswith(".txt"):
        pages = extract_text_from_txt(file_bytes)
    else:
        st.error(f"Unsupported file type: {filename}")
        return -1

    if not pages:
        st.warning(f"No text could be extracted from **{filename}**.")
        return -1

    # ── Chunk ──────────────────────────────────────────────────────────────
    chunks = split_into_chunks(pages, filename)
    if not chunks:
        st.warning(f"Document **{filename}** produced no chunks.")
        return -1

    texts = [c["text"] for c in chunks]

    # ── Embed (batch in 100s to stay within API limits) ────────────────────
    all_embeddings: list[list[float]] = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_embeddings.extend(embed_texts(genai_client, batch))

    # ── Store ──────────────────────────────────────────────────────────────
    store = get_vector_store()
    store.add(
        ids=[c["id"] for c in chunks],
        embeddings=all_embeddings,
        documents=texts,
        metadatas=[
            {"filename": c["filename"], "page": c["page"], "chunk_index": c["chunk_index"]}
            for c in chunks
        ],
    )

    return len(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_relevant_chunks(
    question: str, genai_client: genai.Client, top_k: int = TOP_K
) -> list[dict]:
    """Embed the question and return the top-k most similar stored chunks."""
    store = st.session_state.vector_store
    if store is None or store.count() == 0:
        return []

    q_embedding = embed_texts(genai_client, [question])[0]
    return store.query(q_embedding, n_results=min(top_k, store.count()))


# ─────────────────────────────────────────────────────────────────────────────
# Answer generation
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful, precise document assistant.
Answer the user's question **strictly** using only the provided context chunks below.
If the answer is not contained in the context, say clearly:
"I'm sorry, I could not find an answer to your question in the provided document(s)."

Do NOT make up information or draw on outside knowledge.
When possible, cite the source (filename and page number) at the end of your answer.
"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        header = (
            f"[Chunk {i} | File: {meta.get('filename', 'unknown')} "
            f"| Page: {meta.get('page', '?')} "
            f"| Chunk #: {meta.get('chunk_index', '?')}]"
        )
        context_blocks.append(f"{header}\n{chunk['text']}")

    context_str = "\n\n---\n\n".join(context_blocks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== CONTEXT CHUNKS ===\n{context_str}\n\n"
        f"=== USER QUESTION ===\n{question}\n\n"
        f"=== ANSWER ==="
    )


def generate_answer(question: str, chunks: list[dict], genai_client: genai.Client) -> str:
    """Call Gemini to generate a grounded answer (with automatic retry)."""
    prompt = build_prompt(question, chunks)

    def _call():
        return genai_client.models.generate_content(
            model=CHAT_MODEL, contents=prompt
        ).text

    return call_with_retry(_call)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar UI
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 RAG Assistant")
    st.markdown("---")

    # ── API Key ───────────────────────────────────────────────────────────
    st.subheader("🔑 API Key")
    manual_api_key = st.text_input(
        "Gemini API Key (optional)",
        type="password",
        placeholder="Paste key here if not set in .env / secrets",
        help="Only required if GEMINI_API_KEY is not set as an environment variable or in Streamlit secrets.",
    )

    genai_client = get_genai_client(manual_api_key if manual_api_key else None)

    if genai_client:
        st.success(f"✅ API key loaded from **{st.session_state.api_key_source}**")
    else:
        st.error("❌ No API key found. Please set GEMINI_API_KEY or enter it above.")

    st.markdown("---")

    # ── File Uploader ─────────────────────────────────────────────────────
    st.subheader("📄 Upload Documents")

    if not PYPDF_AVAILABLE:
        st.warning("⚠️ `pypdf` not installed — PDF support disabled. Only `.txt` files are accepted.")

    accepted_types = ["txt"] + (["pdf"] if PYPDF_AVAILABLE else [])
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=accepted_types,
        accept_multiple_files=True,
        help="Upload one or more documents to build the knowledge base.",
    )

    process_btn = st.button(
        "⚙️ Process & Embed Documents",
        disabled=(not uploaded_files or genai_client is None),
        use_container_width=True,
    )

    if process_btn and uploaded_files and genai_client:
        new_files = [
            f for f in uploaded_files
            if f.name not in st.session_state.uploaded_filenames
        ]
        if not new_files:
            st.info("All uploaded files are already processed.")
        else:
            progress_bar = st.progress(0, text="Processing documents…")
            total_new_chunks = 0
            errors = []

            for idx, file in enumerate(new_files):
                progress_bar.progress(
                    idx / len(new_files),
                    text=f"Embedding: {file.name}…",
                )
                try:
                    n = ingest_document(file, genai_client)
                    if n > 0:
                        st.session_state.uploaded_filenames.append(file.name)
                        st.session_state.total_chunks += n
                        total_new_chunks += n
                except Exception as exc:
                    errors.append(f"**{file.name}**: {exc}")
                    logger.exception("Error ingesting %s", file.name)

            progress_bar.progress(1.0, text="Done!")

            if total_new_chunks > 0:
                st.success(
                    f"✅ Added **{total_new_chunks}** chunks from "
                    f"**{len(new_files) - len(errors)}** file(s)."
                )
            for err in errors:
                st.error(err)

    # ── Knowledge base status ─────────────────────────────────────────────
    if st.session_state.uploaded_filenames:
        st.markdown("---")
        st.subheader("📂 Knowledge Base")
        store = st.session_state.vector_store
        st.metric("Total Chunks", store.count() if store else 0)
        for fname in st.session_state.uploaded_filenames:
            st.markdown(f"- 📄 `{fname}`")

    # ── Reset button ──────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🗑️ Clear Documents & Reset Chat", use_container_width=True, type="secondary"):
        reset_store()
        st.success("Session cleared.")
        st.rerun()

    st.markdown("---")
    st.caption(
        "Powered by **Google Gemini** · **Streamlit**\n\n"
        f"Embedding: `{EMBEDDING_MODEL}` | Chat: `{CHAT_MODEL}`"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────────────────────────────────────
st.title("💬 RAG Document Assistant")
st.markdown(
    "Upload your documents in the sidebar, then ask questions about them below. "
    "Answers are grounded exclusively in the uploaded content."
)

# ── Guard rails ───────────────────────────────────────────────────────────────
if genai_client is None:
    st.warning(
        "⚠️ **No API key configured.** "
        "Set `GEMINI_API_KEY` in your `.env` file, Streamlit secrets, or enter it in the sidebar."
    )
    st.stop()

if not st.session_state.uploaded_filenames:
    st.info("📂 No documents loaded yet. Upload and process files using the sidebar to get started.")

# ── Render chat history ───────────────────────────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "citations" in message:
            with st.expander("📎 View Source Citations"):
                for i, chunk in enumerate(message["citations"], start=1):
                    meta = chunk["metadata"]
                    similarity_pct = round((1 - chunk["distance"]) * 100, 1)
                    st.markdown(
                        f"**Chunk {i}** · `{meta.get('filename')}` "
                        f"· Page {meta.get('page')} "
                        f"· Similarity: {similarity_pct}%"
                    )
                    st.text(chunk["text"])
                    if i < len(message["citations"]):
                        st.divider()

# ── Chat input ────────────────────────────────────────────────────────────────
if question := st.chat_input(
    "Ask a question about your document(s)…",
    disabled=not st.session_state.uploaded_filenames,
):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document…"):
            try:
                chunks = retrieve_relevant_chunks(question, genai_client, top_k=TOP_K)

                if not chunks:
                    answer = (
                        "I'm sorry, I could not find any relevant content in the "
                        "uploaded document(s) to answer your question."
                    )
                    st.markdown(answer)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "citations": []}
                    )
                else:
                    answer = generate_answer(question, chunks, genai_client)
                    st.markdown(answer)

                    with st.expander("📎 View Source Citations"):
                        for i, chunk in enumerate(chunks, start=1):
                            meta = chunk["metadata"]
                            similarity_pct = round((1 - chunk["distance"]) * 100, 1)
                            st.markdown(
                                f"**Chunk {i}** · `{meta.get('filename')}` "
                                f"· Page {meta.get('page')} "
                                f"· Similarity: {similarity_pct}%"
                            )
                            st.text(chunk["text"])
                            if i < len(chunks):
                                st.divider()

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "citations": chunks}
                    )

            except Exception as exc:
                err_msg = f"❌ An error occurred: {exc}"
                st.error(err_msg)
                logger.exception("Error generating answer")
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": err_msg, "citations": []}
                )
