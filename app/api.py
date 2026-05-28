from fastapi import FastAPI
from typing import Any, Dict
from app.schemas import QueryRequest, QueryResponse
from rag.pipeline import run_rag

app = FastAPI(title="RAG Evaluation API", version="1.0")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):

    result: Dict[str, Any] = run_rag(question=request.question)
    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        groundedness=result["groundedness"],
        grounded_top1=result["grounded_top1"],
        faithfulness=result["faithfulness"],
        llm_groundedness=result["llm_groundedness"],
    )
