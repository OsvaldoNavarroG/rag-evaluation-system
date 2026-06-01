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

    def retrieve(self, question: str, queries_=None) -> list:
        if not queries_:
            queries = self.query_expander.generate(question)
        else:
            queries = queries_

        all_results = {}

        for q in queries:
            results = self.retriever_fn(q)

            for r in results:
                chunk = r["chunk"]

                # Keep best score if duplicate
                if chunk not in all_results:
                    all_results[chunk] = r
                else:
                    # Optional: keep higher score
                    if r.get("score", 0) > all_results[chunk].get("score", 0):
                        all_results[chunk] = r

        return list(all_results.values())
