import nltk
from nltk.tokenize import sent_tokenize
from typing import List
import numpy as np


def ensure_nltk_resources() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def split_sentences(text) -> List[str]:
    # try nltk first, then switch to spacy
    return sent_tokenize(text)


def load_documents(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def chunk_text(text, chunk_size=50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks


def chunk_text_sentences(text, max_words=50, overlap_sentences=1) -> List[str]:
    sentences: List[str] = split_sentences(text=text)

    chunks = []
    current_chunk: List[str] = []
    current_length = 0

    for _, sentence in enumerate(sentences):
        sentence_length: int = len(sentence.split())

        # if adding this sentence exceeds limit -> finalize chunk
        if current_length + sentence_length > max_words:
            chunks.append(" ".join(current_chunk))

            # overlap: keep last N sentences
            current_chunk = (
                current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            )
            current_length = sum(len(s.split()) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def embed_chunks(chunks: List[str], model) -> np.ndarray:
    embeddings = model.encode(sentences=chunks)
    return np.array(embeddings)


def build_index(embeddings):
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index
