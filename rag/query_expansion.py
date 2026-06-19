from openai import OpenAI
from typing import List
import json

client = OpenAI()


class QueryExpander:
    """
    Generates multiple semantically diverse queries from a single input question.
    """

    def __init__(self, model="gpt-4o-mini", n_queries: int = 3):
        self.model = model
        self.n_queries = n_queries

    def _build_prompt(self, question: str) -> str:
        return f"""
You are helping improve document retrieval.

Generate {self.n_queries} alternative search queries for the question below.
Each query should use different wording or keywords.

Guidelines:
- Keep queries concise
- Use different terminology when possible
- Preserve the original meaning

Question:
{question}

Return ONLY in valid JSON a list of strings with the format:
{{
  "queries": [
    "...",
    "...",
    "..."
  ]
}}
"""

    def generate(self, question: str) -> List[str]:
        prompt = self._build_prompt(question)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content.strip()
            queries = json.loads(content)["queries"]

            # Ensure original query is included
            if question not in queries:
                queries.insert(0, question)
                return queries[: self.n_queries + 1]
            else:
                return queries[: self.n_queries]
        except Exception as e:
            print("[QUERY EXPANSION ERROR]", e)
            return [question]
