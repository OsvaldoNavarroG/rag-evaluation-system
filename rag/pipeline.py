from typing import Any, Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.ingestion import (
    load_documents,
    chunk_text_sentences,
    embed_chunks,
    build_index,
)
from rag.bm25 import BM25Retriever
from rag.hybrid import hybrid_retrieve
from rag.retrieval import dense_retrieve
from rag.reranking import rerank
from rag.generation import generate_answer
from rag.attribution import evaluate_faithfulness, extract_citations
from rag.grounding import is_grounded, is_grounded_top1
from rag.llm_judge import LLMJudge
from rag.multi_query import MultiQueryRetriever
from rag.query_expansion import QueryExpander

# Global model loading
model: SentenceTransformer = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")
judge = LLMJudge()
expander = QueryExpander(n_queries=3)

# Build retrieval system once
text: str = load_documents(path="data/docs.txt")
chunks: List[str] = chunk_text_sentences(text=text)
embeddings: np.ndarray = embed_chunks(chunks=chunks)
index = build_index(embeddings=embeddings)
bm25 = BM25Retriever(chunks=chunks)


def dense_fn(query: str, k: int) -> List[Dict[str, Any]]:
    return dense_retrieve(query=query, index=index, chunks=chunks, model=model, k=k)


def hybrid_fn(query: str):
    return hybrid_retrieve(
        query=query,
        dense_retrieve_fn=dense_fn,
        bm25_retriever=bm25,
        k_dense=5,
        k_bm25=5,
    )


multi_retriever: MultiQueryRetriever = MultiQueryRetriever(
    retriever_fn=hybrid_fn, query_expander=expander
)

# Reusable inference


def run_rag(question: str) -> Dict[str, Any]:
    retrieved: list = multi_retriever.retrieve(question=question)
    reranked: list = rerank(query=question, retrieved_results=retrieved)
    retrieved = reranked[:5]

    retrieved_texts: List[str] = [
        r if isinstance(r, str) else r["chunk"] for r in retrieved
    ]
    answer: str = generate_answer(query=question, context_chunks=retrieved_texts)

    faithfulness_result: Dict[str, bool] = evaluate_faithfulness(
        answer=answer, chunks=retrieved_texts
    )
    citations: List[int] = extract_citations(answer=answer)
    grounded: bool = is_grounded(answer=answer, context_chunks=retrieved_texts)
    grounded_top1: bool = is_grounded_top1(
        answer=answer, context_chunks=retrieved_texts
    )
    llm_eval = judge.evaluate(
        question=question, context_chunks=retrieved_texts, answer=answer
    )

    return {
        "answer": answer,
        "citations": citations,
        "groundedness": grounded,
        "grounded_top1": grounded_top1,
        "faithfulness": faithfulness_result["faithful"],
        "llm_groundedness": llm_eval["grounded"],
        "retrieved_chunks": retrieved_texts,
    }
