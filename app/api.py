from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import Any, Dict
from app.schemas import QueryRequest, QueryResponse
from rag.ingestion import ensure_nltk_resources
from rag.pipeline import run_rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_nltk_resources()
    yield


app = FastAPI(title="RAG Evaluation API", version="1.0", lifespan=lifespan)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try:
        result: Dict[str, Any] = run_rag(question=request.question)
        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"],
            groundedness=result["groundedness"],
            grounded_top1=result["grounded_top1"],
            faithfulness=result["faithfulness"],
            llm_groundedness=result["llm_groundedness"],
            latency=result["latency"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
