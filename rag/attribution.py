import re
from rag.helpers import normalize
from typing import Dict, List, Set

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "in",
    "on",
    "of",
    "to",
    "for",
    "and",
    "or",
    "when",
}


def extract_citations(answer: str) -> List[int]:
    """
    Extract citation indices from answer.

    Supports:
    [1], [2]
    (1), (2)
    """
    bracket_matches: list = re.findall(r"\[(\d+)\]", answer)
    paren_matches: list = re.findall(r"\((\d+)\)", answer)

    citations = bracket_matches + paren_matches

    return [int(c) for c in citations]


def strip_citations(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\(\d+\)", "", text)
    return text


def chunk_supports_answer(answer: str, chunk: str) -> bool:
    clean_answer: str = strip_citations(text=answer)

    answer_words: Set[str] = {
        w for w in normalize(text=clean_answer).split() if w not in STOPWORDS
    }
    chunk_words: Set[str] = {
        w for w in normalize(text=chunk).split() if w not in STOPWORDS
    }

    overlap = len(answer_words & chunk_words)
    coverage = overlap / max(len(answer_words), 1)

    return coverage >= 0.5


def evaluate_faithfulness(answer: str, chunks: List[str]) -> Dict[str, bool]:
    """
    Checks whether cited chunks support the answer.

    Returns:
    {
    "has_citations": bool,
    "valid_citations": bool,
    "faithful": bool
    }
    """
    citations: list = extract_citations(answer=answer)
    if not citations:
        return {"has_citations": False, "valid_citations": False, "faithful": False}

    for idx in citations:
        if idx < 0 or idx >= len(chunks):
            return {"has_citations": True, "valid_citations": False, "faithful": False}

        chunk: str = chunks[idx]

        if not chunk_supports_answer(answer=answer, chunk=chunk):
            return {"has_citations": True, "valid_citations": True, "faithful": False}

    return {"has_citations": True, "valid_citations": True, "faithful": True}
