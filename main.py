from typing import Dict, List
from rag.ingestion import load_documents
from rag.evaluation import compare_chunking_approaches

CONFIGS = {
    "dense": {"use_hybrid": False, "use_rerank": False, "use_multiquery": False},
    "hybrid": {"use_hybrid": True, "use_rerank": False, "use_multiquery": False},
    "hybrid_rerank": {"use_hybrid": True, "use_rerank": True, "use_multiquery": False},
    "multiquery": {"use_hybrid": True, "use_rerank": True, "use_multiquery": True},
}

# 1. Load and prepare data
text: str = load_documents(path="data/docs.txt")

# 2. Test queries
test_data: List[Dict[str, str]] = [
    {
        "question": "What is the main topic of the document?",
        "expected": "machine learning",
    },
    {"question": "What is machine learning?", "expected": "learn from data"},
    {
        "question": "What is deep learning?",
        "expected": "neural networks with many layers",
    },
    {
        "question": "What does speech recognition do?",
        "expected": "spoken language into text",
    },
    {"question": "What is supervised learning?", "expected": "labeled data"},
    {"question": "What is clustering?", "expected": "group similar data points"},
    {
        "question": "What is regression used for?",
        "expected": "predict continuous values",
    },
    {"question": "What is overfitting?", "expected": "learns noise"},
    # medium
    {"question": "What is the role of data preprocessing?", "expected": "cleaning"},
    {
        "question": "What does natural language processing enable?",
        "expected": "understand and generate human language",
    },
    {"question": "What are neural networks inspired by?", "expected": "human brain"},
    {"question": "What does reinforcement learning rely on?", "expected": "rewards"},
    {"question": "What is classification used for?", "expected": "assign labels"},
    {"question": "What is anomaly detection used for?", "expected": "unusual patterns"},
    # hard
    {
        "question": "What is the difference between image processing and image recognition?",
        "expected": "does not necessarily involve learning",
    },
    {
        "question": "How is computer vision different from image recognition?",
        "expected": "broader field",
    },
    # paraphrased
    {
        "question": "How do machines convert spoken words into written text?",
        "expected": "speech recognition",
    },
    {
        "question": "How do systems understand human language?",
        "expected": "natural language processing",
    },
    # indirect, multistep
    {
        "question": "Which models are used for sequence data?",
        "expected": "recurrent neural networks",
    },
    {
        "question": "Which models are especially good for image-related tasks?",
        "expected": "convolutional neural networks",
    },
    {
        "question": "What helps reduce overfitting in models?",
        "expected": "regularization",
    },
    {
        "question": "How do models learn by interacting with an environment?",
        "expected": "reinforcement learning",
    },
]

for name, cfg in CONFIGS.items():
    print(f"\n--- {name.upper()} ---")

    compare_chunking_approaches(text=text, test_data=test_data, **cfg)
