from openai import OpenAI
from dotenv import load_dotenv
from typing import List

load_dotenv()
client = OpenAI()


def generate_answer(query: str, context_chunks: List[str]) -> str:
    numbered_chunks: str = "\n\n".join(
        [f"[{i}] {c}" for i, c in enumerate(context_chunks)]
    )

    prompt = f"""
    You are a precise question-answering system.

    Answer the question using ONLY the provided context.

    Rules:
    - Every statement MUST include a citation [i]
    - Use the chunk index provided
    - Do NOT use external knowledge

    Context:
    {numbered_chunks}

    Question
    {query}

    Answer:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()
