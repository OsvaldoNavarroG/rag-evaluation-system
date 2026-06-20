from typing import Any, Dict, List


class MultiQueryRetriever:
    """
    Expands a query and aggregates retrieval results
    """

    def __init__(self, retriever_fn, query_expander):
        """
        retriever_fn: function(query)-> list of results
        query_expander: QueryExpander instance
        """
        self.retriever_fn = retriever_fn
        self.query_expander = query_expander

    def retrieve(self, expanded_queries: List[str]) -> List[Dict[str, Any]]:
        all_results: Dict[str, Dict[str, Any]] = {}

        for q in expanded_queries:
            results: List[Dict[str, Any]] = self.retriever_fn(q)

            for r in results:
                chunk = r["chunk"]

                # Deduplicate by chunk text while preserving first occurrence
                if chunk not in all_results:
                    all_results[chunk] = r

        return list(all_results.values())
