from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from rag import rag_query_streaming

# Initialise FastAPI app
app = FastAPI(title="SEC 10-K RAG")

# Define request body structure
class QueryRequest(BaseModel):
    question: str
    company: Optional[str] = None

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Query endpoint
@app.post("/query")
def query(request: QueryRequest):
    stream, sources = rag_query_streaming(
        query=request.question,
        company=request.company
    )
    full_answer = "".join(
        chunk.text for chunk in stream if chunk.text
    )
    return {
        "question": request.question,
        "company": request.company,
        "answer": full_answer,
        "sources": sources
    }

