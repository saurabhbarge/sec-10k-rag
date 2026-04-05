from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from rag import rag_query_filtered

# Initialise FastAPI app
app = FastAPI(title="SEC 10-K RAG API")

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
    answer = rag_query_filtered(
        query=request.question,
        company=request.company
    )
    return {
        "question": request.question,
        "company": request.company,
        "answer": answer,
        "sources": answer["sources"]
    }