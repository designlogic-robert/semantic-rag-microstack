import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
DATA_PATH = "data.txt"
INDEX_PATH = "faiss.index"
DOCS_PATH = "docs.npy"


def chunk_text(text: str, max_chars: int = 800):
    """Simple paragraph-based chunker."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    buf = ""

    for p in paragraphs:
        if len(buf) + 1 + len(p) <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p

    if buf:
        chunks.append(buf)

    return chunks


def embed_texts(texts, client: OpenAI) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype="float32")


def main():
    load_dotenv()
    client = OpenAI()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    chunks = chunk_text(raw_text)
    print(f"Chunked into {len(chunks)} pieces")

    vectors = embed_texts(chunks, client)
    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)
    np.save(DOCS_PATH, np.array(chunks, dtype=object))

    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved chunk texts to {DOCS_PATH}")


if __name__ == "__main__":
    main()
