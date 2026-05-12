# DocuMind — Agentic RAG System

An AI document assistant that answers questions strictly from your uploaded documents.
Built with Groq Llama 3.3 70B, HuggingFace embeddings, and FAISS — entirely free to run.

> **Live Demo:** [Add your Streamlit Cloud URL here after deployment]

---

## What it does

- Ingests PDF, TXT, and CSV documents
- Splits them into overlapping chunks and generates vector embeddings locally
- Stores embeddings in a FAISS vector database persisted to disk
- When you ask a question, retrieves the most relevant chunks
- Sends only those chunks to Groq Llama 3.3 70B to generate a grounded answer
- If the answer is not in the documents, it says so — no hallucinations

---

## Tech Stack

| Component     | Technology                        |
|---------------|-----------------------------------|
| LLM           | Groq Llama 3.3 70B (free API)     |
| Embeddings    | HuggingFace all-MiniLM-L6-v2      |
| Vector DB     | FAISS (local, persisted to disk)  |
| UI            | Streamlit (two-tab layout)        |
| Language      | Python 3.11                       |
| Deployment    | Streamlit Cloud (free, public URL)|

---

## Project Structure

```
agentic_rag_system/
├── src/
│   ├── config.py       # all settings — chunk size, model names, API key
│   ├── ingest.py       # loads PDF, TXT, CSV from folder or uploaded bytes
│   ├── chunker.py      # splits documents into 800-char overlapping chunks
│   ├── embedder.py     # HuggingFace embeddings + FAISS index build/load
│   ├── retriever.py    # similarity search with L2 threshold filter
│   ├── agent.py        # RAG agent — retrieval + Groq LLM + memory
│   └── __init__.py
├── data/               # put your documents here (PDF, TXT, CSV)
├── vectorstore/        # FAISS index saved here automatically
├── app.py              # Streamlit web UI (Chat tab + Documents tab)
├── cli.py              # command-line interface
├── build_index.py      # one-time script to build index from data/
├── requirements.txt
├── .env                # GROQ_API_KEY (never committed)
├── .gitignore
└── README.md
```

---

## Architecture

```
Documents (PDF / TXT / CSV)
        |
   [ingest.py]
   Load and parse each file into raw text
        |
   [chunker.py]
   Split into 800-character chunks with 100-character overlap
        |
   [embedder.py]
   HuggingFace MiniLM-L6-v2  ->  384-dimensional float32 vectors
   Saved to FAISS IndexFlatL2 on disk
        |
        +---------------------------+
                                    |
   User types a question            |
        |                           |
   Embed query vector               |
        |                           |
   FAISS similarity search  <-------+
        |
   Filter by L2 threshold (score <= 1.5)
        |
   Top-5 relevant chunks
        |
   [agent.py]
   System prompt + context + conversation history
        |
   Groq Llama 3.3 70B
        |
   Answer — grounded in documents only
```

---

## Local Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/agentic-rag-system.git
cd agentic-rag-system
```

**2. Create a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Create a file called `.env` in the project root:

```
GROQ_API_KEY=your_groq_key_here
```

Get a free key at https://console.groq.com

**5. Add your documents**

Copy your PDF, TXT, or CSV files into the `data/` folder.

**6. Build the vector index**

```bash
python build_index.py
```

This runs once (or whenever you add new documents). You will see progress in the terminal.

**7. Run the web UI**

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

**Or use the CLI**

```bash
python cli.py
```

---

## Deploying to Streamlit Cloud (free public URL)

1. Push this repository to GitHub (make sure `.env` is in `.gitignore`)
2. Go to https://streamlit.io/cloud and sign in with GitHub
3. Click **New app**
4. Select your repository, branch `main`, main file `app.py`
5. Click **Advanced settings** → **Secrets** and add:
   ```
   GROQ_API_KEY = "your_groq_key_here"
   ```
6. Click **Deploy**

Your app will be live at `https://your-app-name.streamlit.app` in about 2 minutes.

---

## How to use the app

**Chat tab**
- Type a question in the text box and click Ask
- The answer appears below with source document tags
- The right panel shows the exact chunks used to generate the answer
- Conversation memory is maintained for follow-up questions

**Documents tab**
- Upload new PDF, TXT, or CSV files using the file uploader
- Click **Add and rebuild knowledge base** to process them
- Or click **Rebuild from data/ folder** if you added files directly to the server
- Current index stats (document count, chunk count) are shown on the right

---

## Requirements

```
streamlit
groq
sentence-transformers
faiss-cpu
PyPDF2
numpy
python-dotenv
```

---

## Key Design Decisions

**RAG over fine-tuning** — Documents can be updated without retraining. Faster and cheaper to build and maintain.

**HuggingFace MiniLM for embeddings** — Runs locally with zero API cost. 384-dimensional vectors are compact and fast to search at this scale.

**FAISS IndexFlatL2** — No server required. Exact nearest-neighbor search. Index persisted to disk and reloaded on startup.

**Similarity threshold filter** — Chunks with L2 distance above 1.5 are discarded before reaching the LLM. This prevents the model from generating answers based on irrelevant context.

**Strict system prompt** — The LLM is explicitly instructed to answer only from the provided context and to return a fixed fallback message if the answer is not present.

---

## Limitations

- Scanned PDFs (image-only) are not supported — PyPDF2 requires a text layer
- The FAISS index is rebuilt from scratch when documents change
- Chunk size (800 chars) is fixed — may not suit all document types
- No cross-encoder re-ranking step
- Conversation memory is session-based and lost on page refresh

---

## Scaling Suggestions

- Replace FAISS with **Pinecone** or **Weaviate** for persistent cloud vector storage
- Add **cross-encoder re-ranking** for better retrieval precision
- Use **async batch embedding** for faster index builds on large document sets
- Add **persistent memory** with SQLite or Redis for cross-session history
- Deploy with **Docker on Google Cloud Run** for autoscaling
- Add **user authentication** for multi-tenant deployments

---

## Bonus Features Implemented

- Conversation memory (last 3 turns maintained per session)
- Agentic retrieval gating (LLM is not called if no relevant chunks are found)
- Web UI with file upload (Streamlit two-tab layout)
- CLI interface
- Public deployment on Streamlit Cloud

---

## License

MIT License — free to use, modify, and distribute.
