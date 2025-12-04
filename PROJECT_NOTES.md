# Semantic RAG Microstack – Design Notes

This repo is a **micro implementation** of the ideas behind Universal Semantic Systems (USS):

- `retriever.py` + `faiss.index` + `docs.npy` model the **Semantic Token space** and fast semantic lookup.
- `llm.py` acts as a minimal **USE-style engine**, consuming context and queries to produce answers.
- `api.py` is a tiny **USR-style execution surface**, exposing a stable interface for higher-level runtimes.

The goal is not to reproduce USS in full, but to show:
- I can actually stand up a concrete service.
- The service already matches the decomposition patterns (ingest → index → retrieve → answer → serve).