import numpy as np
from src.config import TOP_K_RESULTS, SIMILARITY_THRESHOLD
from src.embedder import embed_query


def retrieve(
    query: str,
    index,
    metadata: list[dict],
    top_k: int = TOP_K_RESULTS,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """Return top-k relevant chunks above the similarity threshold."""
    query_vec = embed_query(query)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata) and dist <= threshold:
            results.append({
                "source": metadata[idx]["source"],
                "text":   metadata[idx]["text"],
                "score":  float(dist),
            })

    return results