import json
import time
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


def estimate_input_tokens(message: str) -> int:
    """Estimate input tokens using rule of thumb: 1 token ≈ 4 characters"""
    return len(message) // 4


@app.post("/chat")
async def chat(request: Request, body: dict):
    """Non-streaming chat endpoint"""
    start_time = time.time()
    message = body.get("message", "")
    system_prompt = body.get("system_prompt", "")
    
    input_tokens = estimate_input_tokens(message)
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
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
            result = {
                "response": data.get("message", {}).get("content", ""),
                "model": MODEL_NAME,
                "tokens_used": data.get("eval_count", 0) + data.get("prompt_eval_count", 0),
            }
            
            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                "chat_endpoint",
                endpoint="/chat",
                input_token_estimate=input_tokens,
                latency_ms=round(latency_ms, 2),
                output_tokens=data.get("eval_count", 0),
                model=MODEL_NAME,
            )
            
            return result
    
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(
            "chat_error",
            endpoint="/chat",
            error=str(e),
            latency_ms=round(latency_ms, 2),
            input_token_estimate=input_tokens,
        )
        raise


@app.post("/chat/stream")
async def chat_stream(body: dict):
    """Streaming chat endpoint with Server-Sent Events"""
    start_time = time.time()
    message = body.get("message", "")
    system_prompt = body.get("system_prompt", "")
    
    input_tokens = estimate_input_tokens(message)
    
    async def event_generator():
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
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
                                
                                # Yield only JSON - EventSourceResponse adds "data: " prefix automatically
                                yield json.dumps({"token": token, "done": done})
                                
                                if done:
                                    break
                            except json.JSONDecodeError:
                                # Skip malformed lines
                                continue
                    
                    # Final event with empty token and done=true
                    yield json.dumps({"token": "", "done": True})
            
            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                "chat_stream_endpoint",
                endpoint="/chat/stream",
                input_token_estimate=input_tokens,
                latency_ms=round(latency_ms, 2),
                model=MODEL_NAME,
            )
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                "chat_stream_error",
                endpoint="/chat/stream",
                error=str(e),
                latency_ms=round(latency_ms, 2),
                input_token_estimate=input_tokens,
            )
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
