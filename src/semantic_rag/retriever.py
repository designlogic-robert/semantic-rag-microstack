# src/semantic_rag/retriever.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

import faiss
import numpy as np
from openai import OpenAI

# Load .env from project root
ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ROOT_ENV)

# ---------- Paths & config ----------

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "faiss.index"
DOCS_PATH = BASE_DIR / "docs.npy"

EMBEDDING_MODEL = "text-embedding-3-small"

client = OpenAI()


def embed(texts: List[str]) -> np.ndarray:
    """Get L2-normalized embeddings for a list of texts."""
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    # normalise like we did when indexing
    faiss.normalize_L2(vecs)
    return vecs


class SemanticRetriever:
    def __init__(self, k: int = 4) -> None:
        self.k = k

        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")

        if not DOCS_PATH.exists():
            raise FileNotFoundError(f"Docs file not found at {DOCS_PATH}")

        self.index = faiss.read_index(str(INDEX_PATH))
        # docs.npy is a numpy array of Python strings
        self.docs: List[str] = np.load(DOCS_PATH, allow_pickle=True).tolist()

    def search(self, query: str) -> List[Tuple[str, float]]:
        """Return top-k (chunk_text, score) pairs for the query."""
        q_vec = embed([query])
        scores, indices = self.index.search(q_vec, self.k)
        idxs = indices[0]
        scs = scores[0]

        results: List[Tuple[str, float]] = []
        for i, s in zip(idxs, scs):
            if i < 0:
                continue
            text = self.docs[int(i)]
            results.append((text, float(s)))
        return results
    def retrieve(self, query: str, k: int | None = None) -> List[str]:
        """Return only the chunk texts for the query."""
        # use default k if none provided
        if k is None:
            k = self.k

        # Run the ranked search
        results = self.search(query)

        # Take only the top-k text values
        texts_only = [txt for txt, _ in results[:k]]
        return texts_only



