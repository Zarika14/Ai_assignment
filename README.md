# AI Engineer Take-Home Assignment — Complete Implementation

A complete open-source LLM & Agentic Systems implementation for an insurance company internal tool. The system accepts natural language queries from brokers, retrieves relevant policy documents, reasons over them, and responds — **all powered by open-source models only** (no OpenAI API calls anywhere).

**Status**: Parts 1–5 ✅ **All Complete**

---

## 🏗️ Architecture Overview

```
                        USER QUERY
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼──────┐    ┌──────▼─────┐    ┌──────▼──────┐
    │  PART 3    │    │   PART 1   │    │   PART 2    │
    │   Agent    │    │   Model    │    │     RAG     │
    │  (FastAPI  │───►│   Server   │◄───│  Pipeline   │
    │  port 8001)│    │(FastAPI    │    │             │
    │            │    │ port 8000) │    │ • Chunking  │
    │ Tools:     │    │            │    │ • BGE Embed │
    │ search_    │    │ • /chat    │    │ • FAISS     │
    │ policy     │    │ • /stream  │    │ • Re-rank   │
    │ calc_      │    │ • /health  │    │ • Sources   │
    │ premium    │    │ • /metrics │    └─────────────┘
    │ check_     │    └────────────┘
    │ claim      │
    └──────┬─────┘
           │
    ┌──────▼─────────────┐    ┌────────────────────┐
    │     PART 4         │    │      PART 5         │
    │   Fine-Tuning      │    │  Eval & Observ.     │
    │                    │    │                     │
    │  TinyLlama-1.1B    │    │  • Retrieval hit    │
    │  + LoRA (r=8)      │    │    rate             │
    │  Insurance QA      │    │  • LLM-as-judge     │
    │  54 examples       │    │  • Struct. logging  │
    │  Adapter: ~4MB     │    │  • SQLite metrics   │
    └────────────────────┘    └────────────────────┘
```

---

## 📋 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) running locally with `mistral:latest` pulled
- ~8GB RAM minimum

### Setup

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt
pip install -r rag/requirements.txt
pip install -r model_server/requirements.txt
pip install -r agent/requirements.txt

# 2. Start Ollama (in a separate terminal)
ollama serve
ollama pull mistral:latest

# 3. Build the RAG index (one-time)
python rag/index_documents.py

# 4. Start the model server (separate terminal)
python model_server/main.py
# → http://localhost:8000

# 5. Start the agent server (separate terminal)
python agent/server.py
# → http://localhost:8001

# 6. Run evaluations
python eval/run_evals.py
```

### Or: One-Command Docker Compose

```bash
docker-compose up --build
```

---

## 📁 Repository Structure

```
.
├── model_server/                  # PART 1: Local LLM Serving
│   ├── main.py                    # FastAPI server (chat, stream, health, metrics)
│   ├── test_endpoints.py          # Endpoint tests
│   ├── metrics.db                 # SQLite request log (auto-created)
│   ├── requirements.txt
│   ├── README.md                  # Model choice & config
│   └── Dockerfile
│
├── rag/                           # PART 2: RAG Pipeline
│   ├── pipeline.py                # Complete RAG implementation
│   ├── index_documents.py         # Indexing script (run once)
│   ├── test_comprehensive.py      # Full test suite
│   ├── test_grounded_answer.py    # End-to-end demo
│   ├── test_interactive.py        # Interactive retrieval
│   ├── faiss_index.bin            # Built vector index
│   ├── chunks_metadata.json       # Chunk metadata
│   ├── requirements.txt
│   ├── README.md                  # RAG architecture & chunking rationale
│   └── documents/
│       ├── policy_auto_comprehensive.txt
│       ├── policy_health_bronze.txt
│       └── policy_home_standard.txt
│
├── agent/                         # PART 3: Multi-Turn Agent
│   ├── agent.py                   # Hand-rolled agent loop + 3 tools
│   ├── server.py                  # FastAPI agent endpoint
│   ├── test_agent.py              # Agent test suite
│   ├── requirements.txt
│   └── README.md                  # Agent architecture
│
├── finetune/                      # PART 4: LoRA Fine-Tuning
│   ├── generate_dataset.py        # Generate 54 insurance QA examples
│   ├── train_tiny.py              # LoRA training on TinyLlama-1.1B (CPU)
│   ├── inference_tiny.py          # Inference with fine-tuned adapter
│   ├── inference_compare.py       # Base vs fine-tuned comparison (5 examples)
│   ├── insurance_qa_dataset.json  # 54-sample dataset
│   ├── inference_results.json     # Comparison results
│   ├── training_log.txt           # Training loss log
│   ├── requirements.txt
│   ├── README.md                  # Fine-tuning documentation
│   └── tinyllama_lora_adapter/    # Saved LoRA adapter (~4MB delta only)
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer files...
│
├── eval/                          # PART 5: Evaluation & Observability
│   ├── run_evals.py               # 10-query eval: hit rate + LLM-as-judge
│   └── README.md                  # Eval documentation
│
├── eval_results.json              # Latest evaluation results
├── docker-compose.yml             # One-command orchestration (Bonus)
├── requirements.txt               # Root dependencies
└── README.md                      # This file
```

---

## 🚀 Part 1: Local LLM Setup & Serving

**Model**: Mistral 7B Instruct Q4 via Ollama  
**Endpoints**:
- `GET /health` — Health check
- `POST /chat` — Single-turn non-streaming chat
- `POST /chat/stream` — Token-by-token Server-Sent Events streaming
- `GET /metrics/summary` — **(Bonus)** Aggregated metrics from SQLite

**Performance** (measured on i5-9300H, no GPU):
- Memory: ~4.4GB (Q4 quantization)
- Speed: 2–4 tokens/sec (CPU)
- Latency: ~500ms first token

**Logging**: Every request logs structured JSON (via `structlog`) with timestamp, session_id, input_tokens, latency_ms, output_tokens, tool_calls_made.

See [`model_server/README.md`](model_server/README.md) for full documentation.

> **vLLM Note (Bonus)**: vLLM requires a CUDA GPU and cannot run on this CPU-only machine. Documented with throughput comparison (50–200 tok/s GPU vs 2–4 tok/s CPU) in `model_server/README.md`.

---

## 🔍 Part 2: RAG Pipeline

**3 policy documents**: auto comprehensive, health bronze, home standard  
**Embedding model**: `BAAI/bge-small-en-v1.5` (local, no OpenAI)  
**Vector store**: FAISS `IndexFlatIP`  
**Chunking**: 400-token chunks, 80-token overlap — justified in `rag/README.md`  
**Re-ranker (Bonus)**: `cross-encoder/ms-marco-MiniLM-L-6-v2` — two-stage retrieval  
**Source attribution**: Every answer cites `[filename, chunk N]`

```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline(rerank=True)
pipeline.load_index()  # Load pre-built index

result = pipeline.answer_with_sources("What is the collision deductible?")
print(result["answer"])   # "The collision deductible is $500 [policy_auto_comprehensive.txt, chunk 3]."
print(result["sources"])  # [{"source_file": "policy_auto_comprehensive.txt", "chunk_index": 3, ...}]
```

See [`rag/README.md`](rag/README.md) for full documentation.

---

## 🤖 Part 3: Multi-Turn Agent with Tool Use

**Hand-rolled** — no LangChain, no frameworks. Implements the full agentic loop from scratch.

**3 tools**:
| Tool | What It Does |
|---|---|
| `search_policy(query)` | Queries RAG pipeline from Part 2 |
| `calculate_premium(coverage, risk_score)` | Formula: `coverage * 0.02 * risk_score` |
| `check_claim_status(claim_id)` | Mock claim status from hardcoded dict (CLM-001..CLM-005) |

**Features**:
- ✅ Stateful per-session message history (in-memory dict)
- ✅ JSON tool-call parsing with brace counting (`extract_json_tool_call()`)
- ✅ Graceful handling of malformed JSON (falls back to plain text)
- ✅ 6-turn cap with graceful message on exceeded limit
- ✅ `POST /agent/chat` accepting `{session_id, message}`

```bash
curl -X POST http://localhost:8001/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "broker-1", "message": "What is the collision deductible?"}'
```

See [`agent/README.md`](agent/README.md) for full documentation.

---

## 🧠 Part 4: Fine-Tuning TinyLlama-1.1B

**Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (small model, CPU-trainable)  
**Method**: LoRA — rank=8, alpha=16, target `q_proj`+`v_proj` — only ~0.1% of params are trained  
**Dataset**: 54 hand-crafted insurance Q&A examples across 11 categories  
**Output format**: `{"answer": "...", "confidence": "high|medium|low", "source": "policy|general_knowledge"}`

**Results** (5 comparison examples in `finetune/inference_results.json`):
- Base model: 0/5 valid structured JSON outputs
- Fine-tuned: 4/5 valid structured JSON outputs ✅

**Training details**:
- Loss logged every 5 steps with decreasing trend (see `finetune/training_log.txt`)
- Adapter saved separately: `tinyllama_lora_adapter/adapter_model.safetensors` (~4.3MB — **base model NOT included**)
- Total training time: ~60–90 minutes on CPU

```bash
cd finetune/
python generate_dataset.py   # Generate 54 examples
python train_tiny.py          # Fine-tune (60–90 min on CPU)
python inference_compare.py   # Compare base vs fine-tuned
```

See [`finetune/README.md`](finetune/README.md) for full documentation.

---

## 📊 Part 5: Evaluation & Observability

### Evaluation Results (`eval_results.json`)

| Metric | Result |
|---|---|
| Total queries | 10 |
| **Retrieval hit rate** | **100% (10/10)** |
| **Average LLM-as-judge score** | **5.0 / 5.0** |

```bash
python eval/run_evals.py
```

### Structured Logging

All FastAPI endpoints use `structlog` with `JSONRenderer` — **no print statements**. Every request logs:
- `timestamp` (ISO 8601)
- `session_id`
- `estimated_input_tokens`
- `latency_ms`
- `tool_calls_made`

### Bonus: SQLite Metrics + `/metrics/summary`

Every request is persisted to `model_server/metrics.db`. The endpoint:

```bash
curl http://localhost:8000/metrics/summary
```

Returns average latency, total requests, most-used tools, and endpoint breakdown.

See [`eval/README.md`](eval/README.md) for full documentation.

---

## 🏆 Completion Checklist

### Part 1 — Model Server
- ✅ `model_server/` with README (model choice, quantization, rationale)
- ✅ `POST /chat` (single turn, non-streaming)
- ✅ `POST /chat/stream` (SSE, tokens arrive progressively)
- ✅ Memory footprint documented (4.4GB)
- ✅ Tokens/sec documented (2–4 tok/s CPU)
- ⏭️ **Bonus (vLLM)**: Skipped — no GPU available. Documented with honest throughput comparison.

### Part 2 — RAG Pipeline
- ✅ 3 fake insurance policy documents
- ✅ Local HF embeddings (`BAAI/bge-small-en-v1.5`, no OpenAI)
- ✅ Smart chunking with documented rationale (400 tokens, 80 overlap)
- ✅ FAISS vector store (`IndexFlatIP`)
- ✅ Retrieval returns top-k with similarity scores
- ✅ Grounded answers with source attribution `[file, chunk N]`
- ✅ **Bonus**: Cross-encoder re-ranker (`ms-marco-MiniLM-L-6-v2`)

### Part 3 — Agent
- ✅ Stateful multi-turn agent — **no LangChain**
- ✅ `search_policy(query)` tool (calls RAG pipeline)
- ✅ `calculate_premium(coverage, risk_score)` tool
- ✅ `check_claim_status(claim_id)` tool
- ✅ Hand-rolled loop: history, tool detection, execution, injection
- ✅ Graceful malformed JSON handling
- ✅ 6-turn cap
- ✅ `POST /agent/chat` with per-session history
- ✅ **Bonus (PostgreSQL)**: Optional PostgreSQL persistence via `DATABASE_URL` env var; falls back to in-memory gracefully

### Part 4 — Fine-Tuning
- ✅ Small model: TinyLlama-1.1B-Chat-v1.0
- ✅ 54 training examples (>50 required)
- ✅ Structured JSON output format: `{answer, confidence, source}`
- ✅ LoRA via HuggingFace PEFT (not full fine-tuning)
- ✅ Training loss logged and decreasing (see `training_log.txt`)
- ✅ Adapter weights saved separately (~4.3MB)
- ✅ 5 inference examples: base vs fine-tuned (see `inference_results.json`)
- ✅ **Bonus**: Fine-tuned TinyLlama as agent backbone via `USE_FINETUNED_BACKBONE=true` env var (`agent/finetuned_backbone.py`)

### Part 5 — Evaluation & Observability
- ✅ `eval/run_evals.py` with 10 query-answer pairs
- ✅ Retrieval hit rate reported (100%)
- ✅ LLM-as-judge scoring (avg 5.0/5.0)
- ✅ Structured JSON logging (timestamp, session_id, input_tokens, latency_ms, tool_calls_made)
- ✅ **Bonus**: SQLite metrics table + `GET /metrics/summary` endpoint

### Repo Extras
- ✅ `docker-compose.yml` — one-command Ollama + model server
- ✅ README for all 5 parts

---

## 🧪 Testing

```bash
# Part 1 — Model server endpoints
python model_server/test_endpoints.py

# Part 2 — RAG pipeline
python rag/test_comprehensive.py
python rag/test_grounded_answer.py

# Part 3 — Agent
python agent/test_agent.py

# Part 5 — Evaluations
python eval/run_evals.py
```

---

## 📚 Architecture Decisions

| Component | Choice | Rationale |
|---|---|---|
| LLM (serving) | Mistral 7B Q4 via Ollama | Best quality-speed balance on CPU; Q4 cuts size 6–7× with minimal quality loss |
| Embeddings | BAAI/bge-small-en-v1.5 | Retrieval-optimized, CPU-friendly, no OpenAI dependency |
| Vector store | FAISS IndexFlatIP | Fast exact search, no external service, persistent |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Significant relevance improvement for small latency cost |
| Agent framework | Hand-rolled | Demonstrates understanding of the underlying loop, not just API wrappers |
| Fine-tune model | TinyLlama-1.1B | 7× smaller than Mistral → feasible on CPU; chat variant understands instructions |
| Fine-tune method | LoRA (r=8) | Trains ~0.1% of params; adapter is 4MB vs GB for full fine-tune |
| Metrics store | SQLite | Zero-dependency persistent storage for request analytics |

---

## ⚠️ What Didn't Work / Honest Evaluation

**vLLM swap (Part 1 Bonus)**: vLLM requires CUDA. This machine has an i5-9300H with no GPU. Could not test. Documented throughput difference based on literature (50–200 tok/s GPU vs 2–4 tok/s CPU).

**PostgreSQL session persistence (Part 3 Bonus)**: Implemented with graceful fallback. Set `DATABASE_URL=postgresql://...` to activate. The `agent_sessions` table is created automatically. If Postgres is unavailable, falls back to in-memory dict without crashing.

**Fine-tuned model as agent backbone (Part 4 Bonus)**: Implemented in `agent/finetuned_backbone.py`. Set `USE_FINETUNED_BACKBONE=true` to swap from Ollama → TinyLlama-1.1B + LoRA adapter. TinyLlama is less reliable at tool-call JSON than Mistral-7B, so Ollama remains the recommended default, but the integration works.

**CPU training convergence (Part 4)**: With 3 epochs and 54 examples, the model shows clear improvement (4/5 valid JSON vs 0/5 base) but isn't fully converged. A GPU would allow 10+ epochs and much faster experimentation.
