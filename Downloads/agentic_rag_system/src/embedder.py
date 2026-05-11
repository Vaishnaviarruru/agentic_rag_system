import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL, VECTORSTORE_DIR, VECTORSTORE_PATH

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def build_vector_store(chunks: list[dict]) -> tuple:
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, VECTORSTORE_PATH + ".index")

    metadata = [{"source": c["source"], "text": c["text"]} for c in chunks]
    with open(VECTORSTORE_PATH + "_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    print(f"Vector store saved — {index.ntotal} vectors.")
    return index, metadata


def load_vector_store() -> tuple:
    index = faiss.read_index(VECTORSTORE_PATH + ".index")
    with open(VECTORSTORE_PATH + "_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Vector store loaded — {len(metadata)} chunks.")
    return index, metadata


def vector_store_exists() -> bool:
    return (
        os.path.exists(VECTORSTORE_PATH + ".index")
        and os.path.exists(VECTORSTORE_PATH + "_metadata.json")
    )


def embed_query(query: str) -> np.ndarray:
    model = get_model()
    return model.encode([query], convert_to_numpy=True).astype("float32")