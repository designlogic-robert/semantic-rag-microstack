import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
INDEX_PATH = "faiss.index"
DOCS_PATH = "docs.npy"

_client: OpenAI | None = None
_index = None
_docs = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        load_dotenv()
        _client = OpenAI()
    return _client


def _load_index():
    global _index, _docs
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
        _docs = np.load(DOCS_PATH, allow_pickle=True)
    return _index, _docs


def _embed_query(text: str) -> np.ndarray:
    client = _get_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    return vec.reshape(1, -1)


def retrieve(query: str, k: int = 4):
    index, docs = _load_index()
    q_vec = _embed_query(query)
    distances, indices = index.search(q_vec, k)
    return [docs[i] for i in indices[0]]
