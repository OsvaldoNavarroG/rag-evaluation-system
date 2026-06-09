import json
from openai import OpenAI
from typing import List

client = OpenAI()


class LLMJudge:
    """
    LLM-based evaluator for RAG outputs.

    Evaluates:
    - correctness
    - groundedness (context support)
    """

    def __init__(self, model="gpt-4o-mini"):
        self.model = model

    def _build_prompt(self, question: str, context: str, answer: str):
        return f"""
You are an expert evaluator for question-answering systems.

Given:
- a question
- retrieved context
- an answer

Evaluate the answer on two criteria:

1. Correctness:
Is the answer factually correct with respect to the qusetion?

2. Groundedness:
Is the answer fully supported by the provided context ONLY?
If the answer uses knowledge not present in the context, it is NOT grounded.

---

Question:
{question}

Context:
{context}

Answer:
{answer}

---

Respond ONLY in valid JSON:
{{
"correct":true/false,
"grounded":true/false
}}
"""

    def evaluate(self, question: str, context_chunks: List[str], answer: str) -> dict:
        context: str = "\n\n".join(context_chunks)

        prompt: str = self._build_prompt(
            question=question, context=context, answer=answer
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = response.choices[0].message.content.strip()

            return self._parse_response(content)
        except Exception as e:
            print("[LLM JUDGE ERROR]", e)
            return {"correct": False, "grounded": False}

    def _parse_response(self, content):
        """
        Safely parse JSON output from LLM.
        """
        try:
            data = json.loads(content)
            return {
                "correct": bool(data.get("correct", False)),
                "grounded": bool(data.get("grounded", False)),
            }
        except json.JSONDecodeError:
            print("[PARSE ERROR] Raw response:", content)
            return {"correct": False, "grounded": False}
