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

                # Keep best score if duplicate
                if chunk not in all_results:
                    all_results[chunk] = r
                else:
                    # Optional: keep higher score
                    if r.get("score", 0) > all_results[chunk].get("score", 0):
                        all_results[chunk] = r

        all_results_: list = list(all_results.values())
        all_results_.sort(key=lambda r: r.get("score", 0), reverse=True)
        return all_results_[:10]
