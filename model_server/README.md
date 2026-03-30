# Model Server - Part 1

## Model Choice

**Mistral 7B Instruct Q4** was selected for this assignment because:
- **Size & Performance Balance**: At 7B parameters, Mistral provides excellent reasoning capability while remaining deployable on consumer-grade hardware without GPU acceleration
- **Instruction Following**: The Instruct fine-tuning makes it ideal for chat and agent use cases
- **Open Weights**: Fully open-source and community-supported via Ollama
- **Practical for Development**: Allows rapid prototyping and iteration without cloud GPU costs

## Quantization

**Q4 Quantization** reduces model size and memory footprint using 4-bit weight quantization:
- **Mechanism**: Weights are quantized to 4 bits per parameter instead of 32-bit floats
- **Memory Impact**: Reduces model from ~28GB (fp32) to ~4.4GB (q4), a 6–7x reduction
- **Quality Trade-off**: Minimal impact on output quality for most tasks; token generation speed slightly reduced but acceptable for non-real-time applications
- **Why Q4**: Sweet spot between accuracy and inference speed for CPU-based deployment

## Memory Footprint

- **Measured RAM Usage**: ~4.4GB (as shown by `ollama list`)
- **Ollama Overhead**: Additional ~200–500MB for inference engine
- **Total System Memory**: Minimum 8GB RAM recommended; 16GB+ for comfortable multitasking

## Tokens/Sec

- **Approximate Throughput**: 2–4 tokens/sec on CPU (i5-9300H, no GPU)
- **Hardware**: Intel Core i5-9300H (4 cores, no GPU acceleration)
- **Variance**: Depends on prompt length, message complexity, and system load
- **Suitable For**: Non-real-time workflows (batch processing, async agents, development/testing)

## Endpoints

### 1. `POST /chat` (Non-Streaming)

Sends a single request and waits for complete response.

**Request Body:**
```json
{
  "message": "What is machine learning?",
  "system_prompt": "You are a helpful AI assistant."
}
```

**Response:**
```json
{
  "response": "Machine learning is...",
  "model": "mistral:latest",
  "tokens_used": 150
}
```

**Example curl:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is machine learning?",
    "system_prompt": "You are a helpful AI assistant."
  }'
```

### 2. `POST /chat/stream` (Server-Sent Events)

Streams response token-by-token as they are generated.

**Request Body:**
```json
{
  "message": "Write a short poem about AI",
  "system_prompt": "You are a creative writer."
}
```

**Response (SSE Stream):**
```
data: {"token": "In", "done": false}
data: {"token": " the", "done": false}
data: {"token": " digital", "done": false}
...
data: {"token": "", "done": true}
```

**Example curl:**
```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Write a short poem about AI",
    "system_prompt": "You are a creative writer."
  }' \
  -N
```

(The `-N` flag prevents buffering in curl, allowing you to see tokens as they arrive)

### 3. `GET /health`

Health check to verify model server and Ollama are running.

**Response:**
```json
{
  "status": "ok",
  "model": "mistral:latest"
}
```

**Example curl:**
```bash
curl http://localhost:8000/health
```

## Logging

All endpoints log structured JSON events (via `structlog`) to stdout including:
- **timestamp**: ISO 8601 timestamp
- **endpoint**: Request path (`/chat`, `/chat/stream`, `/health`)
- **session_id**: Session identifier
- **estimated_input_tokens**: Estimated input tokens (len(message) // 4)
- **latency_ms**: Request latency in milliseconds
- **output_tokens**: Actual output token count (from Ollama eval_count)
- **tool_calls_made**: List of tools called (passed from agent)
- **model**: Model name (`mistral:latest`)

Example log output:
```json
{
  "event": "chat_endpoint",
  "endpoint": "/chat",
  "session_id": "user-123",
  "estimated_input_tokens": 45,
  "latency_ms": 2543.45,
  "output_tokens": 128,
  "tool_calls_made": ["search_policy"],
  "model": "mistral:latest",
  "timestamp": "2026-03-28T12:34:56.789Z"
}
```

### SQLite Persistence (Bonus)

In addition to stdout logging, every request is persisted to `metrics.db` (SQLite) in the `model_server/` directory. The database stores:
- All request fields above
- Tool usage counts per tool name
- Request status (success/error)

This enables historical analytics without an external database.

## 4. `GET /metrics/summary` (Bonus)

Returns aggregated metrics from the SQLite request log.

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

**Example curl:**
```bash
curl http://localhost:8000/metrics/summary
```

## vLLM Note (Bonus)

**Why vLLM is not used here:**
- **GPU Requirement**: vLLM requires a CUDA-capable GPU for its PagedAttention optimization
- **Unavailable Hardware**: This machine has no GPU; vLLM would fail to initialize
- **Ollama Alternative**: Ollama provides simpler CPU-based inference without external dependencies

**Throughput Comparison:**
- **vLLM on GPU** (e.g., NVIDIA A100): 50–200 tokens/sec (with continuous batching, PagedAttention)
- **Ollama on CPU** (i5-9300H): 2–4 tokens/sec

**Why the Difference:**
- **PagedAttention**: Reduces memory fragmentation in KV-cache, enabling larger batch sizes and higher GPU utilization
- **Continuous Batching**: Processes multiple requests in parallel without waiting for one to complete
- **GPU Compute**: Matrix operations run 100–1000x faster on specialized hardware
- **CPU Limitation**: Sequential execution, cache limitations, and lack of vector operations result in much lower throughput

**Deployment Path**: For production low-latency requirements, migrate to vLLM on GPU infrastructure.
