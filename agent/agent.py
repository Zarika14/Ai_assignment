#!/usr/bin/env python3
"""
Hand-rolled Multi-Turn Agent for Insurance Queries

Implements:
- Tool use with JSON parsing
- Session management (in-memory OR PostgreSQL)
- Turn counting (max 6 turns)
- Graceful error handling
- No LangChain dependency
- Optional fine-tuned TinyLlama backbone

Environment Variables:
  DATABASE_URL            If set, sessions are persisted to PostgreSQL.
                          Example: postgresql://user:pass@localhost:5432/mydb
                          Falls back to in-memory dict if not set.

  USE_FINETUNED_BACKBONE  If set to "true", uses the local TinyLlama-1.1B
                          + LoRA adapter instead of the Ollama model server.
                          Slower but demonstrates Part 4 integration.

Tools Available:
- search_policy(query): Search insurance documents
- calculate_premium(coverage, risk_score): Calculate premium
- check_claim_status(claim_id): Check claim status
"""

import json
import logging
import os
import requests
from typing import Dict, List, Optional

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ---------------------------------------------------------------------------
# Feature Flags
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
USE_POSTGRES = bool(DATABASE_URL)

USE_FINETUNED_BACKBONE = os.environ.get(
    "USE_FINETUNED_BACKBONE", ""
).lower().strip() == "true"

if USE_POSTGRES:
    logger.info("Session backend: PostgreSQL (%s)", DATABASE_URL.split("@")[-1])
else:
    logger.info("Session backend: in-memory dict")

if USE_FINETUNED_BACKBONE:
    logger.info("LLM backbone: fine-tuned TinyLlama-1.1B + LoRA")
else:
    logger.info("LLM backbone: Ollama model server (HTTP)")


# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def search_policy(query: str) -> str:
    """Search insurance policy documents for relevant information."""
    try:
        import sys
        from pathlib import Path

        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(rerank=False)
        pipeline.load_index()
        results = pipeline.retrieve(query, top_k=3, rerank=False)

        if not results:
            return "No relevant policy information found."

        formatted = "Found relevant policy information:\n\n"
        for i, result in enumerate(results, 1):
            formatted += (
                f"{i}. [{result['source_file']}, chunk {result['chunk_index']}]"
                f" (relevance: {result['similarity_score']:.2%})\n"
                f"   {result['text'][:150]}...\n\n"
            )
        return formatted

    except Exception as e:
        logger.error("search_policy error: %s", e)
        return f"Error searching policies: {str(e)}"


def calculate_premium(coverage: float, risk_score: float) -> str:
    """Calculate annual insurance premium.

    Formula: premium = coverage * base_rate * risk_score
    where base_rate = 0.02 (2% of coverage)
    """
    try:
        if not isinstance(coverage, (int, float)) or coverage <= 0:
            return f"Error: coverage must be a positive number, got {coverage}"
        if not isinstance(risk_score, (int, float)) or risk_score < 0:
            return f"Error: risk_score must be non-negative, got {risk_score}"

        base_rate = 0.02
        annual_premium = coverage * base_rate * risk_score
        monthly_premium = annual_premium / 12

        return (
            f"Premium Calculation:\n"
            f"- Coverage Amount: ${coverage:,.2f}\n"
            f"- Base Rate: {base_rate * 100}%\n"
            f"- Risk Score: {risk_score}x\n"
            f"- Annual Premium: ${annual_premium:,.2f}\n"
            f"- Monthly Premium: ${monthly_premium:,.2f}"
        )

    except Exception as e:
        logger.error("calculate_premium error: %s", e)
        return f"Error calculating premium: {str(e)}"


def check_claim_status(claim_id: str) -> str:
    """Check the status of a claim by ID."""
    claims_db = {
        "CLM-001": {
            "status": "approved",
            "amount": 5000.00,
            "date_filed": "2024-10-15",
            "notes": "Collision damage approved for $5,000",
        },
        "CLM-002": {
            "status": "pending review",
            "amount": 3500.00,
            "date_filed": "2024-11-01",
            "notes": "Under investigation - awaiting adjuster report",
        },
        "CLM-003": {
            "status": "rejected",
            "amount": 0.00,
            "date_filed": "2024-11-03",
            "notes": "Rejected - damage falls under exclusion clause 4.2 (wear and tear)",
        },
        "CLM-004": {
            "status": "in progress",
            "amount": 8200.00,
            "date_filed": "2024-11-10",
            "notes": "Assigned to adjuster Jane Smith - completion expected 2024-12-05",
        },
        "CLM-005": {
            "status": "paid",
            "amount": 4200.00,
            "date_filed": "2024-11-05",
            "date_paid": "2024-11-20",
            "notes": "$4,200 disbursed via check",
        },
    }

    try:
        if claim_id not in claims_db:
            return (
                f"Claim ID '{claim_id}' not found. "
                f"Valid claims: {', '.join(claims_db.keys())}"
            )

        claim = claims_db[claim_id]
        result = (
            f"Claim Status Report:\n"
            f"- Claim ID: {claim_id}\n"
            f"- Status: {claim['status'].upper()}\n"
            f"- Amount: ${claim['amount']:,.2f}\n"
            f"- Date Filed: {claim['date_filed']}"
        )
        if "date_paid" in claim:
            result += f"\n- Date Paid: {claim['date_paid']}"
        result += f"\n- Notes: {claim['notes']}"
        return result

    except Exception as e:
        logger.error("check_claim_status error: %s", e)
        return f"Error checking claim status: {str(e)}"


# ============================================================================
# TOOL REGISTRY
# ============================================================================

TOOLS = {
    "search_policy": search_policy,
    "calculate_premium": calculate_premium,
    "check_claim_status": check_claim_status,
}

SYSTEM_PROMPT = """You are an AI insurance assistant. You help brokers answer questions about policies, calculate premiums, and check claim statuses.

You have access to these tools:
- search_policy(query): Search insurance policy documents for relevant information
- calculate_premium(coverage, risk_score): Calculate annual premium. coverage is a dollar amount (float), risk_score is a multiplier (float, typically 0.5–3.0)
- check_claim_status(claim_id): Check status of a claim by its ID (e.g. CLM-001)

To use a tool, respond with ONLY this exact JSON format and nothing else:
{"tool": "tool_name", "args": {"param1": value1, "param2": value2}}

If you can answer without a tool, respond normally in plain text.
After receiving a tool result, use it to form your final answer to the user.
Be concise and helpful."""


# ============================================================================
# SESSION MANAGEMENT — with optional PostgreSQL persistence
# ============================================================================

# In-memory fallback store
_sessions_memory: Dict[str, List[Dict[str, str]]] = {}


# ---------------------------------------------------------------------------
# PostgreSQL helpers
# ---------------------------------------------------------------------------

def _get_pg_conn():
    """Open a new psycopg2 connection using DATABASE_URL."""
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


def _init_postgres() -> None:
    """Create the sessions table in PostgreSQL if it doesn't exist."""
    try:
        import psycopg2
    except ImportError:
        raise RuntimeError(
            "psycopg2 is required for PostgreSQL persistence.\n"
            "Run: pip install psycopg2-binary"
        )
    conn = _get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    id          SERIAL PRIMARY KEY,
                    session_id  TEXT        NOT NULL,
                    role        TEXT        NOT NULL,
                    content     TEXT        NOT NULL,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_sessions_sid "
                "ON agent_sessions(session_id)"
            )
        conn.commit()
        logger.info("PostgreSQL sessions table ready")
    finally:
        conn.close()


if USE_POSTGRES:
    try:
        _init_postgres()
    except Exception as exc:
        logger.error(
            "PostgreSQL init failed (%s) — falling back to in-memory sessions", exc
        )
        USE_POSTGRES = False


def create_session(session_id: str) -> None:
    """Create a new session (no-op if already exists)."""
    if not USE_POSTGRES:
        if session_id not in _sessions_memory:
            _sessions_memory[session_id] = []
            logger.info("Created in-memory session: %s", session_id)


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    """Return the full message history for a session."""
    if USE_POSTGRES:
        conn = _get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role, content FROM agent_sessions "
                    "WHERE session_id = %s ORDER BY id ASC",
                    (session_id,),
                )
                rows = cur.fetchall()
            return [{"role": row[0], "content": row[1]} for row in rows]
        finally:
            conn.close()
    else:
        if session_id not in _sessions_memory:
            _sessions_memory[session_id] = []
        return _sessions_memory[session_id]


def _append_message(session_id: str, role: str, content: str) -> None:
    """Append one message to a session (in-memory or Postgres)."""
    if USE_POSTGRES:
        conn = _get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agent_sessions (session_id, role, content) "
                    "VALUES (%s, %s, %s)",
                    (session_id, role, content),
                )
            conn.commit()
        finally:
            conn.close()
    else:
        if session_id not in _sessions_memory:
            _sessions_memory[session_id] = []
        _sessions_memory[session_id].append({"role": role, "content": content})


def delete_session(session_id: str) -> bool:
    """Delete a session and all its messages."""
    if USE_POSTGRES:
        conn = _get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM agent_sessions WHERE session_id = %s",
                    (session_id,),
                )
                deleted = cur.rowcount > 0
            conn.commit()
            if deleted:
                logger.info("Deleted Postgres session: %s", session_id)
            return deleted
        finally:
            conn.close()
    else:
        if session_id in _sessions_memory:
            del _sessions_memory[session_id]
            logger.info("Deleted in-memory session: %s", session_id)
            return True
        return False


def clear_all_sessions() -> None:
    """Clear ALL sessions (for testing only)."""
    if USE_POSTGRES:
        conn = _get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM agent_sessions")
            conn.commit()
        finally:
            conn.close()
    else:
        _sessions_memory.clear()
    logger.info("Cleared all sessions")


# ============================================================================
# LLM CALL ABSTRACTION — Ollama HTTP  OR  fine-tuned TinyLlama
# ============================================================================

def _call_llm(
    history_prompt: str,
    model_server_url: str,
    timeout: int = 300,
) -> str:
    """
    Call the configured LLM backend and return the response text.

    If USE_FINETUNED_BACKBONE is True, uses the local TinyLlama + LoRA adapter.
    Otherwise calls the Ollama-backed model server via HTTP.
    """
    if USE_FINETUNED_BACKBONE:
        try:
            from finetuned_backbone import generate_response
            logger.info("Using fine-tuned TinyLlama backbone")
            return generate_response(history_prompt, system_prompt=SYSTEM_PROMPT)
        except Exception as e:
            logger.warning(
                "Fine-tuned backbone failed (%s) — falling back to Ollama", e
            )

    # Default: HTTP call to model server (Ollama)
    response = requests.post(
        f"{model_server_url}/chat",
        json={
            "message": history_prompt,
            "system_prompt": SYSTEM_PROMPT,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"]


# ============================================================================
# AGENT LOOP
# ============================================================================

def run_turn(
    session_id: str,
    message: str,
    model_server_url: str = "http://localhost:8000",
    max_turns: int = 6,
) -> Dict:
    """Execute one turn of the agent loop.

    Args:
        session_id:        Unique session identifier
        message:           User message
        model_server_url:  URL of the model server (used when Ollama backend)
        max_turns:         Maximum number of internal reasoning turns

    Returns:
        Dict with keys: response, tool_calls_made, turns_used, session_id,
                        latency_ms, session_backend, llm_backend
    """
    import time
    start_time = time.time()

    # Ensure session exists
    create_session(session_id)

    # Append user message
    _append_message(session_id, "user", message)

    tool_calls_made = []
    turns_used = 1
    final_response = ""

    while turns_used <= max_turns:
        try:
            # Build history prompt from current DB/memory state
            history = get_session_history(session_id)
            prompt = format_history_for_prompt(history)

            logger.info("[%s] Turn %d: Calling LLM", session_id, turns_used)
            llm_response = _call_llm(prompt, model_server_url)
            logger.info(
                "[%s] Turn %d: Got response (len=%d)",
                session_id, turns_used, len(llm_response),
            )

            # Try to detect a tool call
            parsed = extract_json_tool_call(llm_response)
            if parsed:
                tool_name = parsed.get("tool", "")
                tool_args = parsed.get("args", {})
                logger.info(
                    "[%s] Turn %d: Tool call detected: %s args=%s",
                    session_id, turns_used, tool_name, tool_args,
                )

                if tool_name not in TOOLS:
                    error_msg = (
                        f"Unknown tool: '{tool_name}'. "
                        f"Available: {', '.join(TOOLS.keys())}"
                    )
                    logger.warning("[%s] %s", session_id, error_msg)
                    _append_message(session_id, "assistant", error_msg)
                    final_response = error_msg
                    break

                # Execute tool
                try:
                    tool_func = TOOLS[tool_name]

                    if tool_name == "search_policy":
                        if "query" not in tool_args:
                            raise ValueError("search_policy requires 'query'")
                        tool_result = tool_func(tool_args["query"])

                    elif tool_name == "calculate_premium":
                        if "coverage" not in tool_args or "risk_score" not in tool_args:
                            raise ValueError(
                                "calculate_premium requires 'coverage' and 'risk_score'"
                            )
                        tool_result = tool_func(
                            float(tool_args["coverage"]),
                            float(tool_args["risk_score"]),
                        )

                    elif tool_name == "check_claim_status":
                        if "claim_id" not in tool_args:
                            raise ValueError("check_claim_status requires 'claim_id'")
                        tool_result = tool_func(tool_args["claim_id"])

                    else:
                        tool_result = f"Unknown tool: {tool_name}"

                    tool_calls_made.append(tool_name)
                    logger.info(
                        "[%s] Turn %d: Tool executed: %s (result len=%d)",
                        session_id, turns_used, tool_name, len(tool_result),
                    )

                    # Persist both sides to session
                    _append_message(session_id, "assistant", llm_response)
                    _append_message(
                        session_id, "user", f"Tool result: {tool_result}"
                    )
                    turns_used += 1
                    continue  # next iteration

                except ValueError as e:
                    error_msg = f"Tool argument error: {str(e)}"
                    logger.error("[%s] %s", session_id, error_msg)
                    _append_message(session_id, "assistant", error_msg)
                    final_response = error_msg
                    break

                except Exception as e:
                    error_msg = f"Tool execution error: {str(e)}"
                    logger.error("[%s] %s", session_id, error_msg)
                    _append_message(session_id, "assistant", error_msg)
                    final_response = error_msg
                    break

            else:
                # No tool call — this is the final answer
                _append_message(session_id, "assistant", llm_response)
                final_response = llm_response
                logger.info("[%s] Final plain-text response", session_id)
                break

        except requests.exceptions.Timeout:
            error_msg = "Model server timeout. Please try again."
            logger.error("[%s] %s", session_id, error_msg)
            _append_message(session_id, "assistant", error_msg)
            final_response = error_msg
            break

        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to model server. Is it running?"
            logger.error("[%s] %s", session_id, error_msg)
            _append_message(session_id, "assistant", error_msg)
            final_response = error_msg
            break

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error("[%s] %s", session_id, error_msg)
            _append_message(session_id, "assistant", error_msg)
            final_response = error_msg
            break

    if turns_used > max_turns:
        final_response = (
            "I've reached my turn limit. "
            "Please rephrase your question or break it into smaller parts."
        )
        logger.warning("[%s] Turn limit exceeded: %d", session_id, turns_used)

    latency_ms = (time.time() - start_time) * 1000
    logger.info(
        "[%s] Completed in %.1fms, turns=%d, tools=%d",
        session_id, latency_ms, turns_used, len(tool_calls_made),
    )

    return {
        "response": final_response,
        "tool_calls_made": tool_calls_made,
        "turns_used": turns_used,
        "session_id": session_id,
        "latency_ms": round(latency_ms, 2),
        "session_backend": "postgresql" if USE_POSTGRES else "in-memory",
        "llm_backend": "finetuned-tinyllama" if USE_FINETUNED_BACKBONE else "ollama",
    }


# ============================================================================
# JSON TOOL CALL PARSER
# ============================================================================

def extract_json_tool_call(text: str) -> Optional[Dict]:
    """Extract a JSON tool-call object from mixed text.

    Uses brace-counting so it handles LLM responses that include JSON
    followed by explanation text. Returns None for malformed/missing JSON.
    """
    if '"tool"' not in text:
        return None

    tool_pos = text.find('"tool"')
    start_pos = text.rfind("{", 0, tool_pos)
    if start_pos == -1:
        return None

    brace_count = 0
    end_pos = start_pos
    for i in range(start_pos, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end_pos = i + 1
                break

    try:
        parsed = json.loads(text[start_pos:end_pos])
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


# ============================================================================
# PROMPT FORMATTER
# ============================================================================

def format_history_for_prompt(history: List[Dict[str, str]]) -> str:
    """Format message history into a single prompt string for the LLM."""
    prompt = ""
    for msg in history:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        prompt += f"{role}: {content}\n\n"
    prompt += "ASSISTANT: "
    return prompt
