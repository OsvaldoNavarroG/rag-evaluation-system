from sentence_transformers import CrossEncoder
from typing import List

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, retrieved_results):
    pairs: List[tuple] = [(query, r["chunk"]) for r in retrieved_results]
    scores = reranker.predict(pairs)

    for i, r in enumerate(retrieved_results):
        r["rerank_score"] = float(scores[i])

    return sorted(retrieved_results, key=lambda x: x["rerank_score"], reverse=True)
