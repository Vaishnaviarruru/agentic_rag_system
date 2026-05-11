"""Command-line interface for DocuMind."""
from src.embedder import load_vector_store, vector_store_exists
from src.agent    import ask

def main():
    print("DocuMind CLI")
    print("Type 'exit' to quit\n")

    if not vector_store_exists():
        print("No index found. Run build_index.py first.")
        return

    index, metadata = load_vector_store()
    history = []

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        answer, chunks, history = ask(query, index, metadata, history)
        print(f"\nDocuMind: {answer}")
        if chunks:
            sources = list(set(c["source"] for c in chunks))
            print(f"Sources: {', '.join(sources)}")
        print()

if __name__ == "__main__":
    main()