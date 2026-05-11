    # DocuMind — Agentic RAG System

An AI document assistant that answers questions strictly from your uploaded documents.
No hallucinations — if the answer is not in the documents, it says so.

## Live Demo

[Add your Streamlit Cloud link here after deployment]

## Tech Stack

- LLM: Groq Llama 3.3 70B (free API)
- Embeddings: HuggingFace all-MiniLM-L6-v2 (free, local)
- Vector DB: FAISS
- UI: Streamlit
- Language: Python 3.11

## Project Structure

    agentic_rag_system/
    ├── src/
    │   ├── config.py       # settings and API key
    │   ├── ingest.py       # loads PDF, TXT, CSV
    │   ├── chunker.py      # splits docs into overlapping chunks
    │   ├── embedder.py     # HuggingFace embeddings + FAISS index
    │   ├── retriever.py    # similarity search with threshold
    │   └── agent.py        # RAG agent with Groq LLM + memory
    ├── data/               # put your documents here
    ├── vectorstore/        # FAISS index saved here
    ├── app.py              # Streamlit web UI
    ├── cli.py              # command-line interface
    ├── build_index.py      # one-time index builder
    └── requirements.txt

## Local Setup

    git clone https://github.com/YOUR_USERNAME/documind.git
    cd documind
    pip install -r requirements.txt

    # Add your Groq key to .env
    echo GROQ_API_KEY=your_key_here > .env

    # Add documents to data/ folder, then build index
    python build_index.py

    # Run the app
    streamlit run app.py

## Deploying to Streamlit Cloud (free, public URL)

1. Push this repo to GitHub
2. Go to https://streamlit.io/cloud and sign in with GitHub
3. Click New app, select your repo, set main file to app.py
4. Under Advanced settings add secret:  GROQ_API_KEY = your_key_here
5. Click Deploy — you get a public URL in 2 minutes

## Architecture

    Documents (PDF/TXT/CSV)
         |
    [ingest.py] Load + parse
         |
    [chunker.py] Split into 800-char overlapping chunks
         |
    [embedder.py] HuggingFace MiniLM embeddings -> FAISS index
         |
    User query -> embed -> similarity search -> top-5 chunks
         |
    [agent.py] Groq Llama 3.3 70B + strict system prompt
         |
    Answer (grounded in documents only)

## Limitations

- Scanned PDFs without selectable text will not work
- FAISS index must be rebuilt when documents change
- Retrieval quality depends on chunk size and query phrasing

## Scaling

- Replace FAISS with Pinecone or Weaviate for cloud-hosted persistent vectors
- Add async batch embedding for large document sets
- Add cross-encoder re-ranking for better retrieval precision
- Add user authentication for multi-user deployments