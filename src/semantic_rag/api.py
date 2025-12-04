from fastapi import FastAPI
from pydantic import BaseModel

from .retriever import SemanticRetriever
from .llm import answer_query

app = FastAPI(title="Semantic RAG Microstack")

retriever = SemanticRetriever()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    contexts = retriever.retrieve(req.query, k=req.top_k)
    answer = answer_query(req.query, contexts)
    return QueryResponse(answer=answer)
