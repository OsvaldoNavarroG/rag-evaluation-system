from typing import Any, Dict, List
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
from rag.config import DOC_PATH
from rag.timing import Timer
from rag.ingestion import ensure_nltk_resources

ensure_nltk_resources()


# Global model loading
model: SentenceTransformer = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")
judge = LLMJudge()
expander = QueryExpander(n_queries=3)


class RAGSystem:
    def __init__(self, chunks, model, judge, expander):
        self.chunks = chunks
        self.model = model
        self.judge = judge
        self.expander = expander
        embeddings = embed_chunks(chunks=self.chunks, model=self.model)
        self.index = build_index(embeddings=embeddings)
        self.bm25 = BM25Retriever(chunks=self.chunks)

    def dense(self, query: str, k: int = 5):
        return dense_retrieve(
            query=query, index=self.index, chunks=self.chunks, model=self.model, k=k
        )

    def hybrid(self, query: str):
        return hybrid_retrieve(
            query=query,
            dense_retrieve_fn=self.dense,
            bm25_retriever=self.bm25,
            k_dense=5,
            k_bm25=5,
        )

    def multiquery(self, expanded_queries: List[str]) -> list:
        multi_retriever = MultiQueryRetriever(retriever_fn=self.hybrid)
        return multi_retriever.retrieve(expanded_queries=expanded_queries)

    def query(
        self,
        question: str,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        use_multiquery: bool = True,
    ):
        timer = Timer()
        timer.start("total")

        # retrieval
        if use_multiquery:
            timer.start(name="query_expansion")
            expanded_queries = self.expander.generate(question=question)
            timer.stop(name="query_expansion")
            timer.start(name="retrieval")
            retrieved = self.multiquery(expanded_queries=expanded_queries)
            timer.stop(name="retrieval")
        else:
            timer.start(name="retrieval")
            if use_hybrid:
                retrieved = self.hybrid(question)
            else:
                retrieved = self.dense(question)
            timer.stop(name="retrieval")

        # reranking
        timer.start(name="reranking")
        if use_rerank:
            retrieved = rerank(query=question, retrieved_results=retrieved)[:5]
        timer.stop(name="reranking")
        retrieved_texts = [r if isinstance(r, str) else r["chunk"] for r in retrieved]

        # generation
        timer.start(name="generation")
        answer = generate_answer(query=question, context_chunks=retrieved_texts)
        timer.stop(name="generation")

        # Evaluation
        timer.start(name="evaluation")
        faithfulness_result = evaluate_faithfulness(
            answer=answer, chunks=retrieved_texts
        )
        citations = extract_citations(answer=answer)
        grounded = is_grounded(answer=answer, context_chunks=retrieved_texts)
        grounded_top1 = is_grounded_top1(answer=answer, context_chunks=retrieved_texts)

        timer.stop(name="evaluation")
        timer.stop(name="total")

        return {
            "answer": answer,
            "citations": citations,
            "groundedness": grounded,
            "grounded_top1": grounded_top1,
            "faithfulness": faithfulness_result["faithful"],
            "has_citations": faithfulness_result["has_citations"],
            "retrieved_chunks": retrieved_texts,
            "latency": timer.get(),
        }


# Build retrieval system once
text: str = load_documents(path=DOC_PATH)
chunks: List[str] = chunk_text_sentences(text=text)

default_system: RAGSystem = RAGSystem(
    chunks=chunks, model=model, judge=judge, expander=expander
)


def run_rag(
    question: str,
    use_hybrid: bool = True,
    use_rerank: bool = True,
    use_multiquery: bool = True,
) -> Dict[str, Any]:
    return default_system.query(
        question=question,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
        use_multiquery=use_multiquery,
    )
