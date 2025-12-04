# src/semantic_rag/llm.py
from __future__ import annotations

from pathlib import Path
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
import os

# Load .env from project root
ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ROOT_ENV)

MODEL_NAME = "gpt-4.1-mini"  # or whatever chat model you want

client = OpenAI()


def answer_query(query: str, context_chunks: List[str]) -> str:
    """
    Given a user query and a list of retrieved context chunks,
    call the LLM and return a synthesized answer.
    """

    # Join the chunks into a single context block
    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are an AI assistant helping explain the Universal Semantic Systems (USS) "
        "architecture. Use ONLY the provided context to answer. "
        "If the answer is not in the context, say you don't know."
    )

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"User question:\n{query}\n\n"
        "Answer in clear, straightforward language."
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()
