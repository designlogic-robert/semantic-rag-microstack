from dotenv import load_dotenv
from openai import OpenAI

MODEL = "gpt-4.1-mini"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        load_dotenv()
        _client = OpenAI()
    return _client


def answer(query: str, contexts: list[str]) -> str:
    client = _get_client()
    context_text = "\n\n".join(f"- {c}" for c in contexts)

    prompt = (
        "You are an AI assistant that explains the Universal Semantic Systems "
        "architecture and related concepts clearly.\n\n"
        "Use ONLY the following context snippets. If something is missing, say so "
        "instead of guessing.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Answer in clear, concrete language in 2–4 short paragraphs."
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return resp.choices[0].message.content.strip()
