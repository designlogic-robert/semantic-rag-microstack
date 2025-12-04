# Semantic RAG Microstack

A minimal, **production-style** Retrieval-Augmented Generation (RAG) service.

This repo shows how to:

- Ingest raw text into chunks
- Embed those chunks with OpenAI
- Index them with FAISS
- Expose a clean FastAPI endpoint for semantic questions

It’s intentionally small so you can read it end-to-end in one sitting and still see all the moving parts of a real RAG pipeline.

---

## What this microstack proves

From a founder / hiring-manager perspective, this repo demonstrates that you can:

1. **Design a clean, modular RAG pipeline**
   - Separate ingest, retrieval, and LLM orchestration
   - Use FAISS and OpenAI correctly, without leaking secrets

2. **Ship a small, production-shaped microservice**
   - FastAPI + uvicorn
   - Clear I/O contract (`/query` endpoint, typed request/response models)
   - Artifacts and secrets handled via `.gitignore` and `.env`

3. **Explain the architecture like an engineer, not a vibes coder**
   - Every file has a single, obvious responsibility
   - The stack is easy to extend into a bigger “Semantic Runtime” later

---

## Architecture

**Directories and files:**

```text
semantic-rag-microstack/
├── .gitignore
├── requirements.txt
├── README.md
└── src/
    └── semantic_rag/
        ├── api.py         # FastAPI app and /query endpoint
        ├── data.txt       # Source corpus to index (plain text)
        ├── ingest.py      # Build FAISS index + docs.npy from data.txt
        ├── llm.py         # LLM wrapper for answering with context chunks
        ├── query.py       # CLI helper to test the pipeline
        ├── retriever.py   # FAISS-based semantic retriever
        ├── faiss.index    # (ignored) FAISS index artifact
        └── docs.npy       # (ignored) Chunked text store
```
### High-level flow:

1. #### Ingest
    - `ingest.py` reads `data.txt`
    - Splits into chunks
    - Embeds chunks with `text-embedding-3-small`
    - Stores vectors in `faiss.index`
    - Stores chunk texts in `docs.npy`

2. #### Retrieve
    - `retriever.py` loads `faiss.index` + `docs.npy`
    - Given a query, embeds it and performs vector search
    - Returns top-k chunks (and optionally scores)

3. #### Answer
    - `llm.py` wraps the OpenAI Chat Completions API
    - Given `query` + `context_chunks`, it prompts a chat model (e.g. `gpt-4.1-mini`)
    - Returns a concise, context-grounded answer

4. #### Serve
    - `api.py` exposes `/query` via FastAPI
    - Request: `{ "query": "..." , "top_k": 4 }`
    - Response: `{ "answer": "..." }`

## Setup
1. ### Clone and enter the project
```
git clone https://github.com/<your-username>/semantic-rag-microstack.git
cd semantic-rag-microstack
```
2. ### Create a virtual environment
```
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# or
.venv\Scripts\activate         # Windows
```
3. ### Install dependencies
```
pip install -r requirements.txt
```
4. ### Configure environment variables
Create a `.env` file in the project root (same folder as `README.md`):
```
OPENAI_API_KEY=sk-your-real-key-here
```
The `.env` file is ignored by git via `.gitignore` so your secret key never leaves your machine.

## Usage
### Step 1 – Prepare your data
Edit `src/semantic_rag/data.txt` with whatever corpus you want to index
(e.g. your own architecture notes, product docs, FAQs).

### Step 2 – Build the index
From the project root:
```
cd src/semantic_rag
python ingest.py
cd ../../
```
You should see log output similar to:
```
Chunked into N pieces
Saved index to faiss.index
Saved chunk texts to docs.npy
```
### Step 3 – Run the API
From the project root:

```
uvicorn src.semantic_rag.api:app --reload
```
You should see FastAPI / uvicorn startup logs.

### Step 4 – Send a test query
In another terminal (still in the project root):
```
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Universal Semantic Runtime?"}'
  ```
Example response:
```
{
  "answer": "The Universal Semantic Runtime (USR) is a core layer in the Universal Semantic Systems architecture that handles planning, routing, and governance over semantic operations."
}
```
You can also use the built-in CLI helper:

```
cd src/semantic_rag
python query.py "What is the Universal Semantic Runtime?"
cd ../../
```
## Configuration
You can customize model choices in:
- `retriever.py`
  - `EMBEDDING_MODEL = "text-embedding-3-small"`
- `llm.py`
  - `MODEL_NAME = "gpt-4.1-mini"`

Both modules load the `OPENAI_API_KEY` from `.env` using `python-dotenv`.

## Extending this microstack
Some natural next steps:
- #### Multi-file ingestion
    - Walk a directory, ingest all `.md` / `.txt`files.
- #### Return sources in the API response
    - Include chunk text, index, and score for transparency.
- #### Simple reranking
    - Re-order retrieved chunks with a second pass if needed.
- #### Auth & rate limiting
    - Add API keys or JWT, basic rate limiting, and logging.
- #### Metrics
    - Log queries, response latency, and token usage.

This repo is deliberately small so you can evolve it into:
- A mini semantic runtime for a single product
- A reference microservice inside a larger USS/USR stack

## License
MIT License. Use it, fork it, extend it.