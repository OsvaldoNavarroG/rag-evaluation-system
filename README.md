# Production-Style RAG System with Evaluation & Benchmarking

⏱️ Time to read: ~2 minutes

---

## 🚀 Project Overview

This project builds and evaluates a **production-style Retrieval-Augmented Generation (RAG) system** with:

- Multiple retrieval strategies
- Cross-encoder reranking
- Citation-aware generation
- Faithfulness & hallucination evaluation
- LLM-as-judge benchmarking
- FastAPI deployment
- Docker packaging
- Latency instrumentation
- Experimental benchmarking framework

The project focuses not only on answer quality, but also on:

- retrieval vs generation failures
- ranking quality
- hallucination detection
- evaluation reliability
- latency vs quality tradeoffs

---

# 🔥 Key Results

## Best Performing Configuration

**Hybrid Retrieval + Cross-Encoder Reranking**

Sentence chunking benchmark:

| Metric | Result |
|---|---:|
| Accuracy | **0.91** |
| Retrieval Hit Rate | **0.86** |
| Faithfulness | **0.91** |
| Groundedness | **0.45** |
| Top-1 Groundedness | **0.41** |
| LLM Accuracy | **1.00** |
| LLM Groundedness | **1.00** |

---

# 🧠 Key Findings

## 1. Hybrid + Reranking Improves Quality

Hybrid retrieval improved candidate diversity.

Cross-encoder reranking then improved:

- faithfulness
- grounding
- top-ranked chunk quality

while preserving high recall.

### Sentence Chunking Results

| Config | Accuracy | Faithfulness | Avg Latency |
|---|---:|---:|---:|
| Dense | 0.91 | 0.82 | 1922 ms |
| Hybrid | 0.91 | 0.86 | 2396 ms |
| **Hybrid + Rerank** | **0.91** | **0.91** | **2054 ms** |
| MultiQuery | 0.91 | 0.86 | 3230 ms |

---

## 2. Multi-Query Retrieval Did NOT Help

LLM-based query expansion increased latency without improving quality.

### Observation

| Config | Faithfulness | Latency |
|---|---:|---:|
| Hybrid + Rerank | **0.91** | **2054 ms** |
| MultiQuery | 0.86 | 3230 ms |

### Core Insight

> Multi-query retrieval is only beneficial when the system is recall-limited.

In this project:

- retrieval recall already high (~0.86)
- bottleneck = ranking precision

Adding more queries increased noise and latency.

---

## 3. Evaluation Gap

Traditional heuristic metrics underestimate semantic correctness.

Observed gap:

| Metric | Value |
|---|---:|
| Heuristic Groundedness | ~0.45 |
| LLM Groundedness | ~1.0 |

This motivated:

- LLM-as-judge evaluation
- citation validation
- faithfulness checking

---

# 🏗️ System Architecture

```text
Documents
    ↓
Chunking
    ↓
Dense / Hybrid / MultiQuery Retrieval
    ↓
Cross-Encoder Reranking
    ↓
LLM Generation + Citations
    ↓
Faithfulness + Groundedness Evaluation
    ↓
Latency Instrumentation
    ↓
Benchmark Framework
```

---

# ⚙️ RAG Pipeline

The system uses a reusable **RAGSystem** architecture.

```text
RAGSystem
├── Retrieval
├── Hybrid Search
├── MultiQuery Expansion
├── Reranking
├── Generation
├── Evaluation
└── Latency Tracking
```

This separates:

```text
pipeline.py
→ inference engine

evaluation.py
→ benchmarking

main.py
→ experiment orchestration

FastAPI
→ deployment layer
```

---

# 🔎 Retrieval Strategies Compared

## Dense Retrieval

Semantic vector similarity using FAISS.

Strengths:

- strong semantic recall
- simple baseline

---

## Hybrid Retrieval

Combines:

- dense semantic retrieval
- BM25 lexical retrieval

Improves:

- candidate diversity
- keyword coverage

---

## Hybrid + Reranking (**Best Performing**)

Cross-encoder reranker selects the best chunks.

Improves:

- ranking precision
- answer faithfulness
- grounding

---

## MultiQuery Retrieval

LLM generates alternative query formulations.

Intended to improve recall.

Observed result:

❌ No improvement on this dataset.

---

# 📊 Benchmark Framework

The project includes a configurable benchmark system comparing:

## Retrieval Configurations

- Dense
- Hybrid
- Hybrid + Rerank
- MultiQuery

## Chunking Approaches

- Naive chunking
- Sentence chunking

## Evaluation Metrics

### Heuristic

- Accuracy
- Retrieval Hit Rate
- Groundedness
- Top-1 Groundedness
- Faithfulness
- Citation validation

### LLM-Based

- LLM Accuracy
- LLM Groundedness

### Performance

- Latency instrumentation

---

# ⏱️ Latency Instrumentation

The system measures:

- query expansion
- retrieval
- reranking
- generation
- evaluation
- total latency

Example response:

```json
{
  "latency": {
    "total": 5313.81,
    "query_expansion": 1409.39,
    "retrieval": 106.03,
    "reranking": 224.64,
    "generation": 1575.64,
    "evaluation": 1998
  }
}
```

## Key Insight

Latency analysis revealed:

> Evaluation cost can exceed retrieval + reranking combined.

This highlights a practical production tradeoff:

- online evaluation
vs
- offline evaluation

---

# 🧪 Evaluation Features

The system validates generated answers using:

## Citation Parsing

Supports:

```text
[0]
[1]
(2)
```

---

## Faithfulness

The project evaluates whether generated answers are supported by the cited evidence.

Current implementation:

- Extracts citations from the generated answer
- Verifies citation indices are valid
- Checks whether each cited chunk provides lexical support for the answer

### Limitations

The current implementation uses token-overlap heuristics and is intentionally conservative.

It does **not** yet perform claim-level attribution.

For example:

Answer:
- Claim A [0]
- Claim B [1]

may be marked unfaithful if chunk [0] does not support Claim B, even though Claim B is correctly supported by chunk [1].

Future work includes citation-attribution evaluation to assess whether individual claims are supported by the correct cited chunks.

---

## Groundedness

Two variants:

### Groundedness

Supported by **any retrieved chunk**

### Top-1 Groundedness

Supported by **top-ranked chunk**

Used to evaluate:

- retrieval quality
- ranking quality

## Runtime Diagnostics vs Benchmark Evaluation

The system separates online diagnostics from offline benchmark evaluation.

`RAGSystem.query()` is used by the API and returns lightweight runtime diagnostics:

- groundedness
- top-1 groundedness
- faithfulness
- latency

These metrics are useful for inspecting a single response, but they are not the full benchmark.

`evaluation.py` computes benchmark-level metrics, including:

- heuristic accuracy
- retrieval hit rate
- LLM correctness
- LLM groundedness
- aggregate latency
- configuration comparisons

This separation keeps the API useful for live inspection while keeping experimental evaluation in the benchmark layer.

---

# 🌐 API Deployment

The project includes a deployed API using **FastAPI**.

Run locally:

```bash
uvicorn app.api:app --reload
```

Example request:

```json
POST /query

{
  "question": "What is machine learning?"
}
```

Example response:

```json
{
  "answer": "... [0]",
  "citations": [0],
  "groundedness": true,
  "faithfulness": true,
  "llm_groundedness": true,
  "latency": {
      ...
  }
}
```

---

# 🐳 Docker Support

Containerized deployment supported.

Build:

```bash
docker build -t rag-app .
```

Run:

```bash
docker run -p 8000:8000 rag-app
```

---

# ⚙️ Tech Stack

Core components:

- FAISS
- BM25 (rank-bm25)
- sentence-transformers
- Cross-Encoder reranker
- OpenAI API
- FastAPI
- Docker

Models:

- all-MiniLM-L6-v2
- ms-marco-MiniLM cross-encoder

---

# 🎯 What This Demonstrates

This project demonstrates:

✅ End-to-end RAG engineering  
✅ Retrieval vs ranking analysis  
✅ Hybrid retrieval  
✅ Cross-encoder reranking  
✅ Hallucination detection  
✅ Citation validation  
✅ LLM-as-judge evaluation  
✅ Latency instrumentation  
✅ API deployment  
✅ Docker packaging  
✅ Experimental benchmarking  

Most importantly:

> Understanding not only when advanced techniques work — but when they fail.

---

# 🚀 How To Run

Install:

```bash
pip install sentence-transformers faiss-cpu rank-bm25 openai fastapi uvicorn python-dotenv
```

Run benchmark:

```bash
python main.py
```

Run API:

```bash
uvicorn app.api:app --reload
```

---

# 🔧 Next Improvements

## Planned Improvements

- Citation attribution evaluation
- Claim-level faithfulness checking
- Larger benchmark datasets
- Config-driven experiments