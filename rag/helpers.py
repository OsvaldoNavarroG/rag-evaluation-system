import re
def normalize(text: str) -> str:
    """
    Normalize text for comparison:
    - lowercase
    - collapse whitespace
    - remove punctuation
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()