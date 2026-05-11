import os
import streamlit as st
from src.config   import GROQ_API_KEY
from src.ingest   import load_from_bytes, load_from_folder
from src.chunker  import chunk_documents
from src.embedder import build_vector_store, load_vector_store, vector_store_exists
from src.agent    import ask

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind",
    page_icon=None,
    layout="wide",
)

st.markdown("""
<style>
    /* General */
    .block-container { padding-top: 1.5rem; }
    
    /* Answer card */
    .answer-card {
        background: var(--background-color, #fafafa);
        border-left: 3px solid #333;
        padding: 1rem 1.4rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.95rem;
        line-height: 1.7;
        margin: 0.5rem 0 1rem;
    }

    /* Source pill */
    .source-pill {
        display: inline-block;
        background: #f0f0f0;
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        color: #444;
        margin: 2px 3px;
    }

    /* Status badge */
    .badge-ready {
        background: #d4edda; color: #155724;
        border-radius: 12px; padding: 3px 10px;
        font-size: 0.8rem; font-weight: 500;
    }
    .badge-not-ready {
        background: #fff3cd; color: #856404;
        border-radius: 12px; padding: 3px 10px;
        font-size: 0.8rem; font-weight: 500;
    }

    /* Chunk box */
    .chunk-box {
        background: #f8f8f8;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 0.7rem;
        font-size: 0.8rem;
        color: #555;
        line-height: 1.5;
    }
    div[data-testid="stExpander"] { border: 1px solid #e8e8e8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "index":    None,
    "metadata": None,
    "history":  [],
    "chat_log": [],
    "doc_names":[],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Check API key ─────────────────────────────────────────────────────────────
if not GROQ_API_KEY:
    st.error("Groq API key not found. Add GROQ_API_KEY to your .env file or Streamlit secrets.")
    st.stop()

# ── Auto-load index if it exists (from data/ folder) ─────────────────────────
if st.session_state.index is None and vector_store_exists():
    with st.spinner("Loading existing knowledge base..."):
        st.session_state.index, st.session_state.metadata = load_vector_store()
        existing = load_from_folder()
        st.session_state.doc_names = [d["source"] for d in existing]

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.title("DocuMind")
    st.caption("Agentic RAG system — answers strictly from your documents")
with col_badge:
    st.write("")
    if st.session_state.index is not None:
        n = len(st.session_state.metadata or [])
        st.markdown(f'<span class="badge-ready">Ready — {n} chunks</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-not-ready">No index loaded</span>', unsafe_allow_html=True)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_docs = st.tabs(["Chat", "Documents"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Ask a question")

        query = st.text_input(
            "question",
            placeholder="What does the document say about...",
            label_visibility="collapsed",
        )

        ask_btn = st.button("Ask", type="primary", disabled=(st.session_state.index is None))

        if ask_btn and query.strip():
            with st.spinner("Searching and generating answer..."):
                try:
                    answer, chunks, updated_history = ask(
                        query,
                        st.session_state.index,
                        st.session_state.metadata,
                        st.session_state.history,
                    )
                    st.session_state.history  = updated_history
                    st.session_state.chat_log.append({
                        "question": query,
                        "answer":   answer,
                        "sources":  chunks,
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.index is None:
            st.info("Go to the Documents tab to upload files and build the knowledge base.")

        # ── Conversation history ──────────────────────────────────────────────
        if st.session_state.chat_log:
            st.divider()
            st.subheader("Conversation")

            for entry in reversed(st.session_state.chat_log):
                st.markdown(f"**Q: {entry['question']}**")
                st.markdown(
                    f"<div class='answer-card'>{entry['answer']}</div>",
                    unsafe_allow_html=True,
                )
                if entry["sources"]:
                    unique_sources = list(dict.fromkeys(c["source"] for c in entry["sources"]))
                    pills = " ".join(
                        f"<span class='source-pill'>{s}</span>" for s in unique_sources
                    )
                    st.markdown(f"Sources: {pills}", unsafe_allow_html=True)
                st.write("")

            if st.button("Clear conversation"):
                st.session_state.history  = []
                st.session_state.chat_log = []
                st.rerun()

    with right:
        if st.session_state.chat_log:
            st.subheader("Retrieved context")
            st.caption("Chunks used to answer the last question")

            last = st.session_state.chat_log[-1]
            if last["sources"]:
                for i, chunk in enumerate(last["sources"], 1):
                    with st.expander(f"Chunk {i} — {chunk['source']}  (score {chunk['score']:.3f})"):
                        st.markdown(
                            f"<div class='chunk-box'>{chunk['text'][:500]}{'...' if len(chunk['text']) > 500 else ''}</div>",
                            unsafe_allow_html=True,
                        )
            else:
                st.info("No matching chunks found above the similarity threshold.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.subheader("Manage documents")

    col_up, col_info = st.columns([1, 1])

    with col_up:
        st.write("**Upload new documents**")
        uploaded = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded:
            if st.button("Add and rebuild knowledge base", type="primary"):
                with st.spinner("Processing uploaded files..."):
                    new_docs = []
                    for f in uploaded:
                        doc = load_from_bytes(f.name, f.read())
                        if doc:
                            # Save to data/ so it persists for rebuild
                            os.makedirs("data", exist_ok=True)
                            save_path = os.path.join("data", f.name)
                            with open(save_path, "wb") as out:
                                out.write(f.getvalue() if hasattr(f, "getvalue") else f.read())
                            new_docs.append(doc)
                            st.write(f"Loaded: {f.name}")

                    if new_docs:
                        # Merge with existing data/ docs
                        all_docs = load_from_folder()
                        # Add any uploaded doc not already in folder
                        existing_names = {d["source"] for d in all_docs}
                        for d in new_docs:
                            if d["source"] not in existing_names:
                                all_docs.append(d)

                        chunks = chunk_documents(all_docs)
                        index, metadata = build_vector_store(chunks)
                        st.session_state.index    = index
                        st.session_state.metadata = metadata
                        st.session_state.doc_names = [d["source"] for d in all_docs]
                        st.success(f"Knowledge base built — {len(chunks)} chunks from {len(all_docs)} documents.")
                    else:
                        st.error("No readable content found in uploaded files.")

    with col_info:
        st.write("**Current knowledge base**")

        if st.session_state.index is not None:
            total_chunks = len(st.session_state.metadata or [])
            total_docs   = len(st.session_state.doc_names)
            st.metric("Documents", total_docs)
            st.metric("Chunks indexed", total_chunks)

            if st.session_state.doc_names:
                st.write("**Loaded files:**")
                for name in st.session_state.doc_names:
                    st.write(f"- {name}")
        else:
            st.info("No knowledge base loaded yet. Upload files above.")

    st.divider()

    col_rebuild, col_clear = st.columns([1, 1])

    with col_rebuild:
        st.write("**Rebuild from data/ folder**")
        st.caption("Use this if you added files directly to the data/ folder on the server.")
        if st.button("Rebuild knowledge base from data/"):
            with st.spinner("Rebuilding..."):
                docs = load_from_folder()
                if docs:
                    chunks = chunk_documents(docs)
                    index, metadata = build_vector_store(chunks)
                    st.session_state.index    = index
                    st.session_state.metadata = metadata
                    st.session_state.doc_names = [d["source"] for d in docs]
                    st.success(f"Rebuilt — {len(chunks)} chunks from {len(docs)} documents.")
                else:
                    st.error("No documents found in data/ folder.")

    with col_clear:
        st.write("**Reset**")
        st.caption("Clears the conversation and unloads the knowledge base from memory.")
        if st.button("Clear session", type="secondary"):
            st.session_state.index    = None
            st.session_state.metadata = None
            st.session_state.history  = []
            st.session_state.chat_log = []
            st.session_state.doc_names = []
            st.rerun()