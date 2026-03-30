import json
import time
import sqlite3
import threading
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import structlog
from sse_starlette.sse import EventSourceResponse
import sys

# Configure structured logging to output directly to stdout
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

app = FastAPI(title="Model Server")

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "mistral:latest"

# ---------------------------------------------------------------------------
# SQLite Metrics Store
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "metrics.db"
_db_lock = threading.Lock()


def get_db() -> sqlite3.Connection:
    """Return a thread-local SQLite connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the metrics table if it doesn't exist."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS request_logs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                endpoint      TEXT    NOT NULL,
                session_id    TEXT,
                input_tokens  INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                latency_ms    REAL    DEFAULT 0,
                tool_calls    TEXT    DEFAULT '[]',
                status        TEXT    DEFAULT 'success',
                model         TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_usage (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name   TEXT NOT NULL,
                used_at     TEXT NOT NULL
            )
        """)
        conn.commit()


def log_request(
    endpoint: str,
    session_id: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    tool_calls: list,
    status: str = "success",
) -> None:
    """Insert a request log row into SQLite (thread-safe)."""
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    with _db_lock:
        with get_db() as conn:
            conn.execute(
                """INSERT INTO request_logs
                   (timestamp, endpoint, session_id, input_tokens, output_tokens,
                    latency_ms, tool_calls, status, model)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (now, endpoint, session_id, input_tokens, output_tokens,
                 latency_ms, json.dumps(tool_calls), status, MODEL_NAME),
            )
            # Track individual tool usage
            for tool in tool_calls:
                conn.execute(
                    "INSERT INTO tool_usage (tool_name, used_at) VALUES (?, ?)",
                    (tool, now),
                )
            conn.commit()


# Initialise DB on startup
init_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def estimate_input_tokens(message: str) -> int:
    """Estimate input tokens using rule of thumb: 1 token ≈ 4 characters"""
    return len(message) // 4


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat(request: Request, body: dict):
    """Non-streaming chat endpoint"""
    start_time = time.time()
    message = body.get("message", "")
    system_prompt = body.get("system_prompt", "")
    session_id = body.get("session_id", "unknown")
    tool_calls_made = body.get("tool_calls_made", [])
    if isinstance(tool_calls_made, int):
        tool_calls_made = []

    input_tokens = estimate_input_tokens(message)

    try:
        async with httpx.AsyncClient(timeout=1200.0) as client:
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": message}
                ],
                "stream": False,
            }

            if system_prompt:
                payload["messages"].insert(0, {"role": "system", "content": system_prompt})

            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            output_tokens = data.get("eval_count", 0)
            result = {
                "response": data.get("message", {}).get("content", ""),
                "model": MODEL_NAME,
                "tokens_used": output_tokens + data.get("prompt_eval_count", 0),
            }

            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                "chat_endpoint",
                endpoint="/chat",
                session_id=session_id,
                estimated_input_tokens=input_tokens,
                latency_ms=round(latency_ms, 2),
                output_tokens=output_tokens,
                model=MODEL_NAME,
                tool_calls_made=tool_calls_made,
            )

            log_request("/chat", session_id, input_tokens, output_tokens,
                        round(latency_ms, 2), tool_calls_made)

            return result

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(
            "chat_error",
            endpoint="/chat",
            error=str(e),
            session_id=session_id,
            latency_ms=round(latency_ms, 2),
            estimated_input_tokens=input_tokens,
            tool_calls_made=tool_calls_made,
        )
        log_request("/chat", session_id, input_tokens, 0,
                    round(latency_ms, 2), tool_calls_made, status="error")
        raise


@app.post("/chat/stream")
async def chat_stream(body: dict):
    """Streaming chat endpoint with Server-Sent Events"""
    start_time = time.time()
    message = body.get("message", "")
    system_prompt = body.get("system_prompt", "")
    session_id = body.get("session_id", "unknown")
    tool_calls_made = body.get("tool_calls_made", [])
    if isinstance(tool_calls_made, int):
        tool_calls_made = []

    input_tokens = estimate_input_tokens(message)

    async def event_generator():
        try:
            async with httpx.AsyncClient(timeout=1200.0) as client:
                payload = {
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "user", "content": message}
                    ],
                    "stream": True,
                }

                if system_prompt:
                    payload["messages"].insert(0, {"role": "system", "content": system_prompt})

                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json=payload,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                token = chunk.get("message", {}).get("content", "")
                                done = chunk.get("done", False)

                                yield json.dumps({"token": token, "done": done})

                                if done:
                                    break
                            except json.JSONDecodeError:
                                continue

                    # Final event
                    yield json.dumps({"token": "", "done": True})

            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                "chat_stream_endpoint",
                endpoint="/chat/stream",
                session_id=session_id,
                estimated_input_tokens=input_tokens,
                latency_ms=round(latency_ms, 2),
                model=MODEL_NAME,
                tool_calls_made=tool_calls_made,
            )
            log_request("/chat/stream", session_id, input_tokens, 0,
                        round(latency_ms, 2), tool_calls_made)

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                "chat_stream_error",
                endpoint="/chat/stream",
                error=str(e),
                session_id=session_id,
                latency_ms=round(latency_ms, 2),
                estimated_input_tokens=input_tokens,
                tool_calls_made=tool_calls_made,
            )
            log_request("/chat/stream", session_id, input_tokens, 0,
                        round(latency_ms, 2), tool_calls_made, status="error")
            yield json.dumps({"error": str(e), "done": True})

    return EventSourceResponse(event_generator())


@app.get("/health")
async def health():
    """Health check endpoint"""
    logger.info("health_check", endpoint="/health", model=MODEL_NAME)
    return {
        "status": "ok",
        "model": MODEL_NAME,
    }


@app.get("/metrics/summary")
async def metrics_summary():
    """
    BONUS: Return aggregated metrics from the SQLite log.

    Response includes:
    - total_requests: number of requests logged
    - average_latency_ms: mean latency across all requests
    - most_used_tools: sorted list of tool names and their call counts
    - endpoints: request counts broken down by endpoint
    """
    with _db_lock:
        with get_db() as conn:
            # Total requests
            total = conn.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0]

            # Average latency
            avg_row = conn.execute(
                "SELECT AVG(latency_ms) FROM request_logs"
            ).fetchone()
            avg_latency = round(avg_row[0] or 0.0, 2)

            # Endpoint breakdown
            ep_rows = conn.execute(
                "SELECT endpoint, COUNT(*) as cnt FROM request_logs GROUP BY endpoint ORDER BY cnt DESC"
            ).fetchall()
            endpoints = {row["endpoint"]: row["cnt"] for row in ep_rows}

            # Most-used tools
            tool_rows = conn.execute(
                "SELECT tool_name, COUNT(*) as cnt FROM tool_usage "
                "GROUP BY tool_name ORDER BY cnt DESC"
            ).fetchall()
            most_used_tools = [
                {"tool": row["tool_name"], "count": row["cnt"]}
                for row in tool_rows
            ]

    return {
        "total_requests": total,
        "average_latency_ms": avg_latency,
        "most_used_tools": most_used_tools,
        "endpoints": endpoints,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
