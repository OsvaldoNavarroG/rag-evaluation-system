# RAG Retrieval Optimization & Evaluation

⏱️ **Time to read: ~1 minute**

🔗 **Repository:**
https://github.com/OsvaldoNavarroG/rag-evaluation-project

![Python](https://img.shields.io/badge/Python-3.11-blue)
![RAG](https://img.shields.io/badge/RAG-Evaluation-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)
![BM25](https://img.shields.io/badge/BM25-Hybrid_Retrieval-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-API-success)
![Docker](https://img.shields.io/badge/Docker-Deployment-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-LLM-purple)
![LLM-Judge](https://img.shields.io/badge/LLM-As_Judge-red)

---

# 🏗️ Architecture

```text
                    ┌─────────────────┐
                    │   Documents     │
                    └────────┬────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │ Chunking           │
                  │ Naive / Sentence   │
                  └────────┬───────────┘
                           │
                           ▼
               ┌─────────────────────────┐
               │ Embeddings + BM25 Index │
               │ FAISS + BM25            │
               └──────────┬──────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │ Retrieval            │
                │ Dense / Hybrid       │
                │ Multi-Query          │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Cross-Encoder        │
                │ Reranking            │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ LLM Generation       │
                │ + Citations          │
                └──────────┬───────────┘
                           │
                           ▼
               ┌──────────────────────────┐
               │ Evaluation Layer         │
               │ Groundedness             │
               │ Faithfulness             │
               │ LLM-as-Judge             │
               └──────────┬───────────────┘
                          │
                          ▼
               ┌──────────────────────────┐
               │ FastAPI + Docker         │
               │ Deployable API           │
               └──────────────────────────┘
```

---

# 🚀 Key Results

Evaluated multiple RAG configurations using **heuristic, citation-based, and LLM-as-judge evaluation**.

### Best Results

| Metric                   |    Score |
| ------------------------ | -------: |
| Retrieval Hit Rate       | **0.86** |
| Accuracy                 | **0.91** |
| Groundedness             | **0.91** |
| Top-1 Groundedness       | **0.86** |
| Faithfulness (citations) | **0.82** |
| Citation Rate            | **1.00** |
| LLM Accuracy             | **1.00** |
| LLM Groundedness         | **1.00** |

---

# 🔥 Key Findings

## 1. Sentence Chunking Consistently Improved Quality

Sentence-aware chunking outperformed naive chunking across evaluation runs.

**Best accuracy:**

```text
Sentence chunking → 0.91
Naive chunking → 0.86
```

👉 Chunk boundaries materially affect downstream RAG quality.

---

## 2. Hybrid Retrieval Improved Ranking More Than Recall

Combining:

```text
Dense retrieval (FAISS)
+
BM25 lexical retrieval
+
Cross-encoder reranking
```

produced stronger ranking quality.

Retrieval recall remained high:

```text
~0.86
```

but answer quality and grounding improved.

👉 Bottleneck was **ranking precision**, not retrieval coverage.

---

## 3. Multi-Query Retrieval Did NOT Improve Results

LLM-generated query expansion was evaluated against hybrid reranked retrieval.

| Metric             | Hybrid + Rerank | Multi-Query |
| ------------------ | --------------: | ----------: |
| Retrieval Hit Rate |            0.86 |        0.86 |
| LLM Accuracy       |        **0.91** |   0.86–0.91 |
| LLM Groundedness   |        **0.91** |   0.86–0.91 |

Result:

```text
No recall gain
More candidate noise
No measurable improvement
```

👉 Multi-query is useful primarily when systems are **recall-limited**.

---

## 4. Lexical Metrics Underestimate Semantic Grounding

Three evaluation layers were compared:

| Evaluation Layer | Example Metric |
| ---------------- | -------------- |
| Lexical          | Groundedness   |
| Citation-based   | Faithfulness   |
| Semantic         | LLM Judge      |

Observed:

```text
Faithfulness < Groundedness < LLM Groundedness
```

Meaning:

* lexical metrics are conservative
* semantic support may exist even when exact wording differs
* LLM-based evaluation better captures synthesized answers

👉 Traditional overlap metrics can underestimate semantic correctness.

---

# 📌 Project Overview

This project builds and evaluates a **deployable Retrieval-Augmented Generation (RAG) system** focused on:

* retrieval vs generation error analysis
* hallucination detection
* citation-based reliability
* retrieval strategy comparison
* semantic evaluation
* ranking optimization
* API deployment

---

# 🧠 Pipeline

```text
Documents
→ Chunking (Naive / Sentence)
→ Embeddings
→ FAISS + BM25
→ Hybrid Retrieval
→ Cross-Encoder Reranking
→ LLM Generation + Citations
→ Heuristic + Citation + LLM Evaluation
→ FastAPI Endpoint
→ Docker Deployment
```

---

# ⚙️ Tech Stack

* **FAISS** — dense vector retrieval
* **BM25 (rank-bm25)** — lexical retrieval
* **sentence-transformers** — embeddings + reranker
* **OpenAI API** — generation + LLM evaluation
* **FastAPI** — API layer
* **Docker** — deployment
* **Python / regex evaluation layer** — grounding + citation parsing

---

# 🧪 Retrieval Strategies Compared

## Dense Retrieval

* semantic similarity baseline
* strong recall

## Hybrid Retrieval (BM25 + Dense)

* semantic + keyword retrieval
* improves candidate diversity

## Hybrid + Reranking (**Best Retrieval Strategy**)

* cross-encoder reranks candidates
* improves precision and answer quality

## Multi-Query Retrieval

* LLM-generated query reformulation
* tested for recall improvement

Result:

```text
No measurable benefit
```

because recall was already high.

---

# 🧪 Evaluation Framework

## Heuristic Metrics

* Accuracy
* Retrieval Hit Rate
* Groundedness
* Top-1 Groundedness

## Citation-Based Metrics

* Citation Rate
* Faithfulness

## LLM-Judge Metrics

* LLM Accuracy
* LLM Groundedness

---

# 🌐 API Example

POST `/query`

Request:

```json
{
  "question": "What is machine learning?"
}
```

Response:

```json
{
  "answer": "Machine learning is ... [0]",
  "citations": [0],
  "groundedness": true,
  "grounded_top1": true,
  "faithfulness": true,
  "llm_groundedness": true
}
```

Swagger Docs:

```text
http://localhost:8000/docs
```

---

# 🔍 Core System Insight

```text
If recall is already high:

→ More queries add noise
→ Better ranking adds value
→ Semantic evaluation reveals support missed by lexical metrics
```

---

# 🎯 What This Demonstrates

* End-to-end RAG architecture
* Hybrid retrieval (BM25 + dense)
* Cross-encoder reranking
* Citation parsing + attribution
* Hallucination / groundedness evaluation
* LLM-as-judge evaluation
* Query expansion analysis
* Retrieval vs ranking decomposition
* FastAPI service deployment
* Dockerized ML deployment
* Experimental debugging and metric design

---

# 🚀 Local Run

```bash
pip install -r requirements.txt
uvicorn app.api:app --reload
```

Swagger:

```text
http://localhost:8000/docs
```

---

# 🐳 Docker Deployment

Build:

```bash
docker build -t rag-api .
```

Run:

```bash
docker run -p 8000:8000 --env-file .env rag-api
```

API:

```text
http://localhost:8000/docs
```

---

# 🔧 Future Improvements

* latency instrumentation
* experiment tracking
* weighted hybrid scoring
* semantic citation-faithfulness judge
* larger / noisier datasets
* retrieval benchmarking by query difficulty
