from typing import List
from rag.attribution import chunk_supports_answer


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
