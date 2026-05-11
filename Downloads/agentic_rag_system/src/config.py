import os
import streamlit as st

def get_groq_key():
    # Works both locally (.env) and on Streamlit Cloud (st.secrets)
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv("GROQ_API_KEY", "")

GROQ_API_KEY        = get_groq_key()
LLM_MODEL           = "llama-3.3-70b-versatile"
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"

CHUNK_SIZE          = 800
CHUNK_OVERLAP       = 100
TOP_K_RESULTS       = 5
SIMILARITY_THRESHOLD = 1.5        # FAISS L2 distance — lower = more similar

DATA_DIR            = "data"
VECTORSTORE_DIR     = "vectorstore"
VECTORSTORE_PATH    = os.path.join(VECTORSTORE_DIR, "faiss_index")