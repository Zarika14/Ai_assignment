#!/usr/bin/env python3
"""
FastAPI Server for Multi-Turn Agent

Endpoints:
- POST /agent/chat  - Send message and get response
- GET  /agent/sessions/{session_id} - Get session history
- DELETE /agent/sessions/{session_id} - Clear session
- GET  /agent/health - Health check (shows session + LLM backend)

Environment Variables:
  DATABASE_URL           PostgreSQL URL for persistent sessions
  USE_FINETUNED_BACKBONE Set to 'true' to use TinyLlama + LoRA adapter
"""

import json
import time
from fastapi import FastAPI, Request
import structlog
from agent import (
    run_turn, get_session_history, delete_session, create_session,
    USE_POSTGRES, USE_FINETUNED_BACKBONE
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
app = FastAPI(title="Insurance Agent Server")


@app.post("/agent/chat")
async def agent_chat(request: Request, body: dict):
    """Multi-turn agent chat endpoint.
    
    Request body:
    {
        "session_id": "user-123",
        "message": "What is the collision deductible?"
    }
    
    Response:
    {
        "response": "Based on the policy...",
        "tool_calls_made": ["search_policy"],
        "turns_used": 2,
        "session_id": "user-123",
        "latency_ms": 1234.5
    }
    """
    start_time = time.time()
    
    try:
        session_id = body.get("session_id", "")
        message = body.get("message", "").strip()
        
        # Validate inputs
        if not session_id:
            error_response = {
                "error": "session_id is required",
                "response": "",
                "tool_calls_made": [],
                "turns_used": 0,
                "session_id": ""
            }
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                "agent_chat_error",
                error="missing_session_id",
                latency_ms=round(latency_ms, 2)
            )
            return error_response
        
        if not message:
            error_response = {
                "error": "message is required and cannot be empty",
                "response": "",
                "tool_calls_made": [],
                "turns_used": 0,
                "session_id": session_id
            }
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                "agent_chat_error",
                error="empty_message",
                session_id=session_id,
                latency_ms=round(latency_ms, 2)
            )
            return error_response
        
        # Token estimation
        input_tokens = len(message) // 4
        
        # Run agent turn
        result = run_turn(session_id, message)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Log request — all 5 required fields per assignment spec
        logger.info(
            "agent_chat",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            session_id=session_id,
            input_tokens=input_tokens,
            latency_ms=round(latency_ms, 2),
            turns_used=result["turns_used"],
            tools_called=len(result["tool_calls_made"]),
            tool_calls_made=result["tool_calls_made"]
        )
        
        return result
    
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(
            "agent_chat_error",
            error=str(e),
            latency_ms=round(latency_ms, 2)
        )
        return {
            "error": str(e),
            "response": f"Error: {str(e)}",
            "tool_calls_made": [],
            "turns_used": 0,
            "session_id": body.get("session_id", "")
        }


@app.get("/agent/sessions/{session_id}")
async def get_session(session_id: str):
    """Get the full message history for a session.
    
    Response:
    {
        "session_id": "user-123",
        "history": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ],
        "turn_count": 5
    }
    """
    try:
        create_session(session_id)  # Ensure session exists
        history = get_session_history(session_id)
        
        logger.info(
            "get_session",
            session_id=session_id,
            turn_count=len(history)
        )
        
        return {
            "session_id": session_id,
            "history": history,
            "turn_count": len(history)
        }
    
    except Exception as e:
        logger.error(
            "get_session_error",
            session_id=session_id,
            error=str(e)
        )
        return {
            "error": str(e),
            "session_id": session_id,
            "history": [],
            "turn_count": 0
        }


@app.delete("/agent/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    """Delete a session (clear history).
    
    Response:
    {
        "session_id": "user-123",
        "deleted": true
    }
    """
    try:
        deleted = delete_session(session_id)
        
        logger.info(
            "delete_session",
            session_id=session_id,
            deleted=deleted
        )
        
        return {
            "session_id": session_id,
            "deleted": deleted
        }
    
    except Exception as e:
        logger.error(
            "delete_session_error",
            session_id=session_id,
            error=str(e)
        )
        return {
            "error": str(e),
            "session_id": session_id,
            "deleted": False
        }


@app.get("/agent/health")
async def agent_health():
    """Health check endpoint — also reports which backends are active."""
    logger.info("agent_health_check")
    return {
        "status": "ok",
        "service": "insurance-agent",
        "version": "1.0",
        "session_backend": "postgresql" if USE_POSTGRES else "in-memory",
        "llm_backend": "finetuned-tinyllama" if USE_FINETUNED_BACKBONE else "ollama",
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Insurance Agent API",
        "endpoints": {
            "POST /agent/chat": "Send message to agent",
            "GET /agent/sessions/{session_id}": "Get session history",
            "DELETE /agent/sessions/{session_id}": "Delete session",
            "GET /agent/health": "Health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
