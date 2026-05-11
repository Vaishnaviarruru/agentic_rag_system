"""
Run this once to build the FAISS index from files in data/.
After this, the Streamlit app loads the index automatically.

Usage:
    python build_index.py
"""
from src.ingest   import load_all_documents
from src.chunker  import chunk_documents
from src.embedder import build_vector_store

if __name__ == "__main__":
    docs   = load_all_documents()
    if not docs:
        print("No documents found in data/ — add some files and retry.")
    else:
        chunks = chunk_documents(docs)
        build_vector_store(chunks)
        print("Index built successfully.")