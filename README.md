# Semantic RAG Microstack

A lightweight Retrieval-Augmented Generation pipeline designed to demonstrate semantic architecture principles from USS: UST, USR, USE, SynCE, SCP, ORCH-C, and the Semantic Token + Posture model.

## Why This Exists

Most RAG demos are shallow. This microstack shows how your higher-order architecture ideas behave in an actual working retrieval pipeline:

- UST (Universal Semantic Token Model)
- USR (Universal Semantic Runtime)
- USE (Universal Semantic Engine)
- SynCE (Synthetic Cognitive Engine)
- SCP (Semantic Control Protocol)
- ORCH-C (Deterministic Planner)
- Semantic Tokens + Posture System

Founders evaluating AI engineers get a concrete demonstration of:

- architecture-level understanding  
- semantic modeling  
- retrieval pipelines  
- vector indexing  
- API engineering  
- clean, deterministic reasoning  

## Features

- Local FAISS vector index  
- Simple ingestion pipeline (data.txt → FAISS)  
- LLM integration (OpenAI or any API-compatible model)  
- FastAPI server with a single `/query` endpoint  
- Retrieval + synthesis pipeline  
- Explains USS architecture back to the user  

## File Structure

```
├── data.txt
├── ingest.py
├── retriever.py
├── llm.py
├── api.py
├── requirements.txt
└── README.md
```
## Installation

### 1. Clone the Repo
```bash
git clone https://github.com/<your-username>/semantic-rag-microstack
cd semantic-rag-microstack
```
### 2. Create & Activate Virtual Environment
```
python -m venv .venv
source .venv/Scripts/activate
```
### 3. Install Dependencies
```
pip install -r requirements.txt
```
### 4. Add Your OpenAI Key
Create a file named `.env`:
```
OPENAI_API_KEY=your_key_here
```
### Build the Vector Index
```
python ingest.py
```
This reads `data.txt`, chunks it, embeds it, builds a FAISS index, and writes `faiss.index`.

### Run the API
```
uvicorn api:app --reload
```
Server runs at:
```
http://127.0.0.1:8000/query
```

### Query the Model
POST JSON:
```
{
  "query": "Explain ORCH-C in simple terms."
}
```
Example response:
```
{
  "answer": "ORCH-C is the deterministic planner inside USS. It routes meaning with posture-aware constraints."
}
```
### What This Demonstrates to Founders
This repo shows that you can:
1. Build functional AI pipelines
2. Work with embeddings, vector search, and RAG
3. Structure clean, modular Python services
4. Explain complex systems clearly
5. Deliver production-ready microservices

### Extending the Microstack
You can add:
- PDF/Markdown ingestion
- Multi-file retrieval
- Query decomposition
- Structured output formats
- Agent-style orchestration
- Caching or reranking

### License
MIT License.

### Author
Robert Hansen
- Chief Semantic Architect - Universal Semantic Systems
- GitHub: https://github.com/designlogic-robert
- LinkedIn: https://linkedin.com/in/roberthansen-ai

