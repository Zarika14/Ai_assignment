# Part 5: Evaluation & Observability

## Overview

This module measures how well the RAG pipeline works and provides structured observability for the entire system. As the assignment spec says: **"Don't just build it — measure it."**

---

## 📁 Files

```
eval/
├── README.md          # This file
├── requirements.txt   # Dependencies
└── run_evals.py       # Main evaluation script
```

---

## 🧪 Evaluation Script — `run_evals.py`

### What It Does

1. **Retrieval Hit Rate** — For each of 10 test queries, checks whether the correct source document appears in the top-3 retrieved chunks
2. **Answer Generation** — Runs each query through the full RAG pipeline (retrieve → LLM generate)
3. **LLM-as-Judge Scoring** — Sends `(query, reference_answer, generated_answer)` to the model server and asks it to score 1–5 how well the generated answer matches the reference

### Test Dataset (10 queries)

| # | Query | Expected Source |
|---|---|---|
| 1 | What is the collision deductible? | policy_auto_comprehensive.txt |
| 2 | Does the auto policy cover intentional damage? | policy_auto_comprehensive.txt |
| 3 | When are renewal documents mailed for the auto policy? | policy_auto_comprehensive.txt |
| 4 | What is the individual annual deductible for the Bronze plan? | policy_health_bronze.txt |
| 5 | Is prior authorization required for an MRI? | policy_health_bronze.txt |
| 6 | How much is the copay for a Tier 1 generic drug? | policy_health_bronze.txt |
| 7 | What is the limit for personal property coverage? | policy_home_standard.txt |
| 8 | Does the standard homeowners policy cover flood damage? | policy_home_standard.txt |
| 9 | What is the additional living expenses coverage limit? | policy_home_standard.txt |
| 10 | How many quotes from licensed contractors are recommended for a home claim? | policy_home_standard.txt |

All three policy documents are covered (auto, health, home).

---

## 🚀 How to Run

### Prerequisites

1. Ollama must be running:
   ```bash
   ollama serve
   ```

2. Model server must be running:
   ```bash
   python model_server/main.py
   ```

3. RAG index must be built:
   ```bash
   python rag/index_documents.py
   ```

### Run Evaluations

```bash
# From the project root
python eval/run_evals.py
```

Or from the `eval/` directory:
```bash
cd eval/
python run_evals.py
```

**Expected time**: 10–30 minutes (each query makes two LLM calls: answer generation + judge scoring)

---

## 📊 Metrics Reported

### 1. Retrieval Hit Rate
```
hit_rate = (queries where correct source is in top-3 retrieved chunks) / total_queries
```
- **1.0** = perfect retrieval (correct document always retrieved)
- **0.7** = 7 out of 10 queries retrieved the right document

### 2. LLM-as-Judge Score
Each answer is scored 1–5 by calling the model server with the prompt:
```
"Score the given Answer based on how well it matches the Reference and answers the Query.
Output ONLY a single integer from 1 to 5.
1=Completely incorrect/irrelevant, 5=Perfectly matches reference."
```
The `average_score` is the mean across all 10 queries.

---

## 📋 Output

### Console Output
```
[1/10] Evaluated query: 'What is the collision deductible?'
  Hit: True, Score: 5
[2/10] Evaluated query: 'Does the auto policy cover intentional damage?'
  Hit: True, Score: 5
...
--- FINAL METRICS (JSON) ---
{
  "summary": {
    "total_queries": 10,
    "hit_rate": 1.0,
    "average_score": 4.8
  },
  ...
}
```

### Results File (`eval_results.json`)
Full results including per-query details:
```json
{
  "summary": {
    "total_queries": 10,
    "hit_rate": 1.0,
    "average_score": 5.0
  },
  "details": [
    {
      "query": "What is the collision deductible?",
      "expected_source": "policy_auto_comprehensive.txt",
      "retrieved_sources": ["policy_auto_comprehensive.txt", ...],
      "is_hit": true,
      "reference_answer": "The collision deductible is $500.",
      "generated_answer": "The collision deductible is $500 [policy_auto_comprehensive.txt, chunk 3].",
      "score": 5,
      "retrieval_latency_ms": 1393.95,
      "generation_latency_ms": 267242.22
    },
    ...
  ]
}
```

---

## ✅ Actual Results (from `eval_results.json`)

| Metric | Value |
|---|---|
| Total queries | 10 |
| **Retrieval hit rate** | **100% (10/10)** |
| **Average LLM-as-judge score** | **5.0 / 5.0** |

The RAG pipeline achieved **perfect retrieval** across all 10 test queries, always returning the correct source document in the top-3 results. The LLM-as-judge rated every generated answer a **5/5** — exactly matching the reference answers.

---

## 🔍 Observability (Structured Logging)

Every request to the FastAPI model server and agent server is logged as structured JSON — **not print statements**.

### Model Server Logging (`model_server/main.py`)

Uses `structlog` with `JSONRenderer`. Every `/chat` and `/chat/stream` request logs:

```json
{
  "event": "chat_endpoint",
  "endpoint": "/chat",
  "session_id": "eval-run-1",
  "estimated_input_tokens": 45,
  "latency_ms": 2543.12,
  "output_tokens": 128,
  "model": "mistral:latest",
  "tool_calls_made": 0,
  "timestamp": "2026-03-29T14:30:00.123456Z"
}
```

### Agent Server Logging (`agent/server.py`)

Every `/agent/chat` request logs:

```json
{
  "event": "agent_chat",
  "session_id": "user-123",
  "input_tokens": 15,
  "latency_ms": 5234.67,
  "turns_used": 3,
  "tools_called": 2,
  "tool_calls_made": ["search_policy", "calculate_premium"],
  "timestamp": "2026-03-29T14:35:00.456789Z"
}
```

### Required Fields (All Present ✅)

| Field | Present In |
|---|---|
| `timestamp` | Both servers (ISO 8601) |
| `session_id` | Both servers |
| `estimated_input_tokens` | Both servers |
| `latency_ms` | Both servers |
| `tool_calls_made` | Agent server |

---

## 🏆 Bonus: SQLite Metrics & `/metrics/summary` Endpoint

The model server also includes a **SQLite-backed metrics store** and a `GET /metrics/summary` endpoint:

```bash
# Start model server
python model_server/main.py

# Check metrics summary
curl http://localhost:8000/metrics/summary
```

**Response:**
```json
{
  "total_requests": 42,
  "average_latency_ms": 3421.5,
  "most_used_tools": [
    {"tool": "search_policy", "count": 18},
    {"tool": "calculate_premium", "count": 12},
    {"tool": "check_claim_status", "count": 7}
  ],
  "endpoints": {
    "/chat": 25,
    "/chat/stream": 10,
    "/agent/chat": 7
  }
}
```

---

## 📝 Design Decisions

### Why 10 Evaluation Queries?
The assignment requires "at least 10" — we chose exactly 10, distributed across all three policy documents (3 auto, 3 health, 4 home) to ensure all documents are tested.

### Why LLM-as-Judge?
Automated string matching would penalize paraphrased but correct answers. An LLM judge can assess semantic similarity and nuance, giving a more accurate quality signal even without human annotation.

### Scoring Scale (1–5)
- **5**: Answer is factually correct and directly answers the question
- **4**: Answer is mostly correct with minor omissions
- **3**: Partially correct but missing key information
- **2**: Weak relevance, significant errors
- **1**: Completely wrong or irrelevant

### Retrieval Hit Rate Definition
A "hit" is when the **expected source document filename** appears in the top-3 retrieved chunk sources. This tests whether the retrieval system surfaces the right document, regardless of which specific chunk.
