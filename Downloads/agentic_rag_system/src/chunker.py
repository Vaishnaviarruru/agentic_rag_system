from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_document(doc: dict) -> list[dict]:
    """Split a single document into overlapping chunks."""
    text   = doc["content"]
    source = doc["source"]
    chunks = []
    start  = 0
    idx    = 0

    while start < len(text):
        end   = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if len(chunk) > 30:          # skip tiny fragments
            chunks.append({
                "source":   source,
                "chunk_id": idx,
                "text":     chunk,
            })
            idx += 1

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk a list of documents and return all chunks."""
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
        print(f"  {doc['source']} -> {len(doc_chunks)} chunks")

    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks