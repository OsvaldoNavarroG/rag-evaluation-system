from rag.attribution import evaluate_faithfulness
from rag.grounding import is_grounded, is_grounded_top1
from rag.ingestion import chunk_text_sentences
from rag.pipeline import RAGSystem, model, judge, expander
from typing import Any, Dict, List


def evaluate_answer(predicted, expected):
    predicted = predicted.lower()
    expected = expected.lower()

    return expected in predicted


def run_pipeline(
    chunking_fn,
    text: str,
    test_data,
    label,
    use_hybrid: bool,
    use_rerank: bool,
    use_multiquery: bool,
) -> List[Dict[str, Any]]:
    print(f"\n===== {label} =====")

    chunks: List[str] = chunking_fn(text)
    system = RAGSystem(chunks=chunks, model=model, judge=judge, expander=expander)

    results = []

    for item in test_data:
        query = item["question"]
        expected = item["expected"]

        result: Dict[str, Any] = system.query(
            question=query,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
            use_multiquery=use_multiquery,
        )
        answer: str = result["answer"]
        retrieved_texts = result["retrieved_chunks"]

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
                "latency_ms": result["latency"]["total"],
            }
        )

    return results


def compare_chunking_approaches(
    text: str, test_data: List[Dict[str, str]], **rag_kwargs
):
    from rag.ingestion import chunk_text as naive_chunk_text

    naive_results: List[dict] = run_pipeline(
        chunking_fn=naive_chunk_text,
        text=text,
        test_data=test_data,
        label="Naive Chunking",
        **rag_kwargs,
    )
    sentence_results: List[dict] = run_pipeline(
        chunking_fn=chunk_text_sentences,
        text=text,
        test_data=test_data,
        label="Sentence Chunking",
        **rag_kwargs,
    )
    print("\n===== SUMMARY =====")
    print("Naive:", summarize(results=naive_results))
    print("Sentence:", summarize(results=sentence_results))


def summarize(results: List[dict]) -> dict:
    total: int = len(results)
    correct: int = sum(r["correct"] for r in results)
    hits: int = sum(r["retrieval_hit"] for r in results)
    grounded: int = sum(r["grounded"] for r in results)
    latency: int = sum(r["latency_ms"] for r in results)
    return {
        "accuracy": correct / total,
        "retrieval_hit_rate": hits / total,
        "groundedness": grounded / total,
        "grounded_top1": sum(r["grounded_top1"] for r in results) / total,
        "llm_accuracy": sum(r["llm_correct"] for r in results) / total,
        "llm_groundedness": sum(r["llm_grounded"] for r in results) / total,
        "faithfulness": sum(r["faithful"] for r in results) / total,
        "citation_rate": sum(r["has_citations"] for r in results) / total,
        "avg_latency_ms": latency / total,
    }
