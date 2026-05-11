from groq import Groq
from src.config import GROQ_API_KEY, LLM_MODEL
from src.retriever import retrieve

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are DocuMind, an intelligent document assistant.

Your rules — follow them strictly:
1. Answer ONLY using the context provided below.
2. If the answer is not in the context, respond with exactly:
   "I don't have enough information in the provided documents to answer this."
3. Never use general knowledge or make up information.
4. Always cite which document (source) your answer comes from.
5. Be clear, accurate, and concise."""


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def ask(
    query: str,
    index,
    metadata: list[dict],
    history: list[dict] | None = None,
) -> tuple[str, list[dict], list[dict]]:
    """
    Run a RAG query.
    Returns: (answer, retrieved_chunks, updated_history)
    """
    retrieved = retrieve(query, index, metadata)

    if not retrieved:
        answer = "I don't have enough information in the provided documents to answer this."
        updated = (history or []) + [
            {"role": "user",      "content": query},
            {"role": "assistant", "content": answer},
        ]
        return answer, [], updated

    context = format_context(retrieved)

    user_message = f"""Context from documents:
{context}

---
Question: {query}

Answer using only the context above:"""

    messages = list(history or [])[-6:]          # keep last 3 turns
    messages.append({"role": "user", "content": user_message})

    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=full_messages,
        max_tokens=1024,
        temperature=0.1,
    )

    answer = response.choices[0].message.content.strip()

    updated_history = list(history or []) + [
        {"role": "user",      "content": query},
        {"role": "assistant", "content": answer},
    ]

    return answer, retrieved, updated_history