import numpy as np
from rag.helpers import normalize
from rag.attribution import evaluate_faithfulness, chunk_supports_answer
from rag.ingestion import chunk_text_sentences, embed_chunks, build_index
from rag.bm25 import BM25Retriever
from rag.hybrid import hybrid_retrieve
from rag.retrieval import dense_retrieve
from rag.reranking import rerank
from rag.generation import generate_answer
from rag.llm_judge import LLMJudge
from rag.multi_query import MultiQueryRetriever
from rag.query_expansion import QueryExpander
from typing import Any, Dict, List
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
judge = LLMJudge()
expander = QueryExpander(n_queries=3)


def is_grounded(answer: str, context_chunks: List[str]) -> bool:
    """
    Checks wether the answer is supported by ANY retrieved chunk.
    """
    for chunk in context_chunks:
        if chunk_supports_answer(answer=answer, chunk=chunk):
            return True

    return False


def is_grounded_top1(answer: str, context_chunks: List[str]) -> bool:
    """
    Checks if the answer is supported by the TOP retrieved chunk only.
    """
    if not context_chunks:
        return False

    return chunk_supports_answer(answer=answer, chunk=context_chunks[0])


def evaluate_answer(predicted, expected):
    predicted = predicted.lower()
    expected = expected.lower()

    return expected in predicted


def run_pipeline(chunking_fn, text, test_data, label) -> List[Dict[str, Any]]:
    print(f"\n===== {label} =====")

    chunks: List[str] = chunking_fn(text)
    embeddings: np.ndarray = embed_chunks(chunks=chunks)
    index = build_index(embeddings=embeddings)

    bm25 = BM25Retriever(chunks=chunks)

    def dense_fn(query: str, k: int):
        return dense_retrieve(query=query, index=index, chunks=chunks, model=model, k=k)

    def hybrid_fn(query: str):
        return hybrid_retrieve(
            query=query,
            dense_retrieve_fn=dense_fn,
            bm25_retriever=bm25,
            k_dense=5,
            k_bm25=5,
        )

    multi_retriever = MultiQueryRetriever(
        retriever_fn=hybrid_fn, query_expander=expander
    )

    results = []

    for item in test_data:
        query = item["question"]
        expected = item["expected"]

        # baseline retrieval
        # retrieved: list = retrieve(query=query, index=index, chunks=chunks, k=3)
        # retrieved: list = dense_retrieve(
        #     query=query, index=index, chunks=chunks, model=model, k=5
        # )

        # reranked retrieval
        # retrieved: list = hybrid_retrieve(
        #     query=query,
        #     dense_retrieve_fn=dense_fn,
        #     bm25_retriever=bm25,
        #     k_dense=5,
        #     k_bm25=5,
        # )

        retrieved: list = multi_retriever.retrieve(question=query)
        reranked: list = rerank(query=query, retrieved_results=retrieved)
        retrieved = reranked[:5]

        retrieved_texts = [r if isinstance(r, str) else r["chunk"] for r in retrieved]
        answer = generate_answer(query=query, context_chunks=retrieved_texts)

        faithfulness_result = evaluate_faithfulness(answer, retrieved_texts)
        faithful = faithfulness_result["faithful"]
        has_citations = faithfulness_result["has_citations"]

        # Metrics
        is_correct = evaluate_answer(predicted=answer, expected=expected)
        hit = any(expected.lower() in c.lower() for c in retrieved_texts)
        grounded: bool = is_grounded(answer=answer, context_chunks=retrieved_texts)
        grounded_top1: bool = is_grounded_top1(
            answer=answer, context_chunks=retrieved_texts
        )

        # evaluate with LLMJudge
        llm_eval: dict = judge.evaluate(
            question=query, context_chunks=retrieved_texts, answer=answer
        )
        llm_correct = llm_eval["correct"]
        llm_grounded = llm_eval["grounded"]

        # debug snippet
        # if llm_grounded and not grounded:
        #     print("\n[SEMANTIC GROUNDING DETECTED]")
        #     print("Q:", query)
        #     print("Answer:", answer)

        # print("\n[HYBRID RETRIEVAL]")
        # for r in retrieved:
        #     print(f"{r['source']}: {r['chunk'][:100]}")
        # expanded_queries = expander.generate(question=query)
        # print("\n[MULTI-QUERY]")
        # for q in expanded_queries:
        #     print("-", q)
        if not faithful:
            print("\n[UNFAITHFUL ANSWER]")
            print("Q:", query)
            print("Answer:", answer)
            print("Chunks:")
            for i, c in enumerate(retrieved_texts):
                print(f"[{i}]", c[:120])

        results.append(
            {
                "question": query,
                "correct": is_correct,
                "retrieval_hit": hit,
                "grounded": grounded,
                "grounded_top1": grounded_top1,
                "llm_correct": llm_correct,
                "llm_grounded": llm_grounded,
                "faithful": faithful,
                "has_citations": has_citations,
            }
        )

    return results


def compare_chunking_approaches(text: str, test_data: List[Dict[str, str]]):
    from rag.ingestion import chunk_text as naive_chunk_text

    naive_results: List[dict] = run_pipeline(
        chunking_fn=naive_chunk_text,
        text=text,
        test_data=test_data,
        label="Naive Chunking",
    )
    sentence_results: List[dict] = run_pipeline(
        chunking_fn=chunk_text_sentences,
        text=text,
        test_data=test_data,
        label="Sentence Chunking",
    )
    print("\n===== SUMMARY =====")
    print("Naive:", summarize(results=naive_results))
    print("Sentence:", summarize(results=sentence_results))


def summarize(results: List[dict]) -> dict:
    total: int = len(results)
    correct: int = sum(r["correct"] for r in results)
    hits: int = sum(r["retrieval_hit"] for r in results)
    grounded: int = sum(r["grounded"] for r in results)

    return {
        "accuracy": correct / total,
        "retrieval_hit_rate": hits / total,
        "groundedness": grounded / total,
        "grounded_top1": sum(r["grounded_top1"] for r in results) / total,
        "llm_accuracy": sum(r["llm_correct"] for r in results) / total,
        "llm_groundedness": sum(r["llm_grounded"] for r in results) / total,
        "faithfulness": sum(r["faithful"] for r in results) / total,
        "citation_rate": sum(r["has_citations"] for r in results) / total,
    }
