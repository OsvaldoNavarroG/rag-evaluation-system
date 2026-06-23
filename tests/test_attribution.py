from rag.attribution import extract_citations, evaluate_citation_precision

def test_duplicate_citations_are_counted_once() -> None:
    answer = "Claim A [0]. Claim B [0][1]."

    citations = extract_citations(answer=answer)
    unique_citations = list(dict.fromkeys(citations))

    assert citations == [0,0,1]
    assert unique_citations == [0,1]

def test_citation_precision_uses_unique_indices(monkeypatch) -> None:
    answer = "Claim A [0]. Claim B [0][1]."
    chunks = ["supporting chunk", "unsupported chunk"]
    support = {
        "supporting chunk": True,
        "unsupported chunk": False
    }

    monkeypatch.setattr(
        "rag.attribution.chunk_supports_answer",
        lambda answer, chunk: support[chunk]
    )

    result = evaluate_citation_precision(answer=answer, chunks=chunks)

    assert result["citation_precision"] == 0.5