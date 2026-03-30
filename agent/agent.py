#!/usr/bin/env python3
"""
Hand-rolled Multi-Turn Agent for Insurance Queries

Implements:
- Tool use with JSON parsing
- Session management
- Turn counting (max 6 turns)
- Graceful error handling
- No LangChain dependency

Tools Available:
- search_policy(query): Search insurance documents
- calculate_premium(coverage, risk_score): Calculate premium
- check_claim_status(claim_id): Check claim status
"""

import json
import logging
import requests
from typing import Dict, List, Optional

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def search_policy(query: str) -> str:
    """Search insurance policy documents for relevant information.
    
    Args:
        query: Search query about insurance policies
        
    Returns:
        Formatted string with top 3 chunks and their sources
    """
    try:
        # Import RAG pipeline with proper path handling
        import sys
        from pathlib import Path
        
        # Add parent directory to path (Ai_assignment folder)
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        from rag.pipeline import RAGPipeline
        
        # Initialize pipeline
        pipeline = RAGPipeline(rerank=False)
        pipeline.load_index()
        
        # Retrieve relevant chunks
        results = pipeline.retrieve(query, top_k=3, rerank=False)
        
        if not results:
            return "No relevant policy information found."
        
        # Format results
        formatted = "Found relevant policy information:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. [{result['source_file']}, chunk {result['chunk_index']}] (relevance: {result['similarity_score']:.2%})\n"
            formatted += f"   {result['text'][:150]}...\n\n"
        
        return formatted
    
    except Exception as e:
        logger.error(f"search_policy error: {e}")
        return f"Error searching policies: {str(e)}"


def calculate_premium(coverage: float, risk_score: float) -> str:
    """Calculate annual insurance premium.
    
    Formula: premium = coverage * base_rate * risk_score
    where base_rate = 0.02 (2% of coverage)
    
    Args:
        coverage: Coverage amount in dollars (float)
        risk_score: Risk multiplier, typically 0.5–3.0 (float)
        
    Returns:
        Formatted string with premium breakdown
    """
    try:
        # Validate inputs
        if not isinstance(coverage, (int, float)) or coverage <= 0:
            return f"Error: coverage must be a positive number, got {coverage}"
        
        if not isinstance(risk_score, (int, float)) or risk_score < 0:
            return f"Error: risk_score must be non-negative, got {risk_score}"
        
        # Calculate premium
        base_rate = 0.02  # 2% of coverage
        annual_premium = coverage * base_rate * risk_score
        monthly_premium = annual_premium / 12
        
        # Format response
        result = f"""Premium Calculation:
- Coverage Amount: ${coverage:,.2f}
- Base Rate: {base_rate * 100}%
- Risk Score: {risk_score}x
- Annual Premium: ${annual_premium:,.2f}
- Monthly Premium: ${monthly_premium:,.2f}"""
        
        return result
    
    except Exception as e:
        logger.error(f"calculate_premium error: {e}")
        return f"Error calculating premium: {str(e)}"


def check_claim_status(claim_id: str) -> str:
    """Check the status of a claim by ID.
    
    Args:
        claim_id: Claim ID (e.g., CLM-001)
        
    Returns:
        Formatted string with claim status and details
    """
    # Hardcoded claim data
    claims_db = {
        "CLM-001": {
            "status": "approved",
            "amount": 5000.00,
            "date_filed": "2024-10-15",
            "notes": "Collision damage approved for $5,000"
        },
        "CLM-002": {
            "status": "pending review",
            "amount": 3500.00,
            "date_filed": "2024-11-01",
            "notes": "Under investigation - awaiting adjuster report"
        },
        "CLM-003": {
            "status": "rejected",
            "amount": 0.00,
            "date_filed": "2024-11-03",
            "notes": "Rejected - damage falls under exclusion clause 4.2 (wear and tear)"
        },
        "CLM-004": {
            "status": "in progress",
            "amount": 8200.00,
            "date_filed": "2024-11-10",
            "notes": "Assigned to adjuster Jane Smith - completion expected 2024-12-05"
        },
        "CLM-005": {
            "status": "paid",
            "amount": 4200.00,
            "date_filed": "2024-11-05",
            "date_paid": "2024-11-20",
            "notes": "$4,200 disbursed via check"
        }
    }
    
    try:
        # Look up claim
        if claim_id not in claims_db:
            return f"Claim ID '{claim_id}' not found. Valid claims: " + ", ".join(claims_db.keys())
        
        claim = claims_db[claim_id]
        
        # Format response
        result = f"""Claim Status Report:
- Claim ID: {claim_id}
- Status: {claim['status'].upper()}
- Amount: ${claim['amount']:,.2f}
- Date Filed: {claim['date_filed']}"""
        
        if "date_paid" in claim:
            result += f"\n- Date Paid: {claim['date_paid']}"
        
        result += f"\n- Notes: {claim['notes']}"
        
        return result
    
    except Exception as e:
        logger.error(f"check_claim_status error: {e}")
        return f"Error checking claim status: {str(e)}"


# ============================================================================
# TOOL REGISTRY
# ============================================================================

TOOLS = {
    "search_policy": search_policy,
    "calculate_premium": calculate_premium,
    "check_claim_status": check_claim_status,
}

# Tool schema (plain English for LLM prompt)
TOOL_DESCRIPTIONS = """
You have access to these tools:

1. search_policy(query: string)
   - Searches insurance policy documents for relevant information
   - Arguments: query - the search question (e.g., "What is the deductible?")
   - Returns information from relevant policy sections

2. calculate_premium(coverage: number, risk_score: number)
   - Calculates annual insurance premium
   - Arguments: 
     * coverage - dollar amount of coverage (e.g., 100000)
     * risk_score - risk multiplier, typically 0.5 to 3.0 (e.g., 1.2)
   - Returns annual and monthly premium amounts with breakdown

3. check_claim_status(claim_id: string)
   - Checks the status of an insurance claim
   - Arguments: claim_id - the claim ID (e.g., "CLM-001")
   - Returns claim status, amount, and details
"""

# System prompt (exact content as required)
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
# SESSION MANAGEMENT
# ============================================================================

# In-memory session storage: {session_id: [{"role": "user/assistant", "content": "..."}, ...]}
sessions: Dict[str, List[Dict[str, str]]] = {}


def create_session(session_id: str) -> None:
    """Create a new session."""
    if session_id not in sessions:
        sessions[session_id] = []
        logger.info(f"Created session: {session_id}")


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    """Get the full message history for a session."""
    if session_id not in sessions:
        create_session(session_id)
    return sessions[session_id]


def delete_session(session_id: str) -> bool:
    """Delete a session."""
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"Deleted session: {session_id}")
        return True
    return False


def clear_all_sessions() -> None:
    """Clear all sessions (for testing)."""
    sessions.clear()
    logger.info("Cleared all sessions")


# ============================================================================
# AGENT LOOP
# ============================================================================

def run_turn(
    session_id: str,
    message: str,
    model_server_url: str = "http://localhost:8000",
    max_turns: int = 6
) -> Dict:
    """Execute one turn of the agent loop.
    
    Args:
        session_id: Unique session identifier
        message: User message
        model_server_url: URL of the model server
        max_turns: Maximum number of turns allowed
        
    Returns:
        Dict with keys:
            - response: Final answer to the user
            - tool_calls_made: List of tools called
            - turns_used: Number of turns used
            - session_id: Session ID
            - error: Error message (if any)
    """
    import time
    start_time = time.time()
    
    # Create session if needed
    if session_id not in sessions:
        create_session(session_id)
    
    history = get_session_history(session_id)
    tool_calls_made = []
    turns_used = 0
    final_response = ""
    
    # Add user message to history
    history.append({"role": "user", "content": message})
    turns_used = 1
    
    # Agentic loop (max 6 turns)
    while turns_used <= max_turns:
        try:
            # Prepare messages for model server
            messages_for_model = history.copy()
            
            # Call model server with full history
            logger.info(f"[{session_id}] Turn {turns_used}: Calling model server")
            response = requests.post(
                f"{model_server_url}/chat",
                json={
                    "message": format_history_for_prompt(messages_for_model),
                    "system_prompt": SYSTEM_PROMPT
                },
                timeout=300
            )
            response.raise_for_status()
            
            llm_response = response.json()["response"]
            logger.info(f"[{session_id}] Turn {turns_used}: Got response (length={len(llm_response)})")
            
            # Try to parse as tool call
            tool_call_detected = False
            try:
                # Extract JSON from response (handles mixed JSON + plain text)
                parsed = extract_json_tool_call(llm_response)
                
                if parsed:
                    json_str = str(parsed)
                    logger.info(f"[{session_id}] Turn {turns_used}: JSON tool call extracted successfully")
                    tool_call_detected = True
                    tool_name = parsed.get("tool", "")
                    tool_args = parsed.get("args", {})
                    
                    logger.info(f"[{session_id}] Turn {turns_used}: Tool call DETECTED: {tool_name} with args {tool_args}")
                    
                    # Validate tool exists
                    if tool_name not in TOOLS:
                        error_msg = f"Unknown tool: '{tool_name}'. Available tools: {', '.join(TOOLS.keys())}"
                        logger.warning(f"[{session_id}] {error_msg}")
                        history.append({"role": "assistant", "content": error_msg})
                        final_response = error_msg
                        break
                    
                    # Validate and execute tool
                    try:
                        tool_func = TOOLS[tool_name]
                        
                        # Call tool with provided args
                        if tool_name == "search_policy":
                            if "query" not in tool_args:
                                raise ValueError("search_policy requires 'query' argument")
                            tool_result = tool_func(tool_args["query"])
                        
                        elif tool_name == "calculate_premium":
                            if "coverage" not in tool_args or "risk_score" not in tool_args:
                                raise ValueError("calculate_premium requires 'coverage' and 'risk_score' arguments")
                            tool_result = tool_func(
                                float(tool_args["coverage"]),
                                float(tool_args["risk_score"])
                            )
                        
                        elif tool_name == "check_claim_status":
                            if "claim_id" not in tool_args:
                                raise ValueError("check_claim_status requires 'claim_id' argument")
                            tool_result = tool_func(tool_args["claim_id"])
                        
                        else:
                            tool_result = f"Unknown tool: {tool_name}"
                        
                        tool_calls_made.append(tool_name)
                        logger.info(f"[{session_id}] Turn {turns_used}: Tool EXECUTED successfully: {tool_name}, result length={len(tool_result)}")
                        
                        # Append tool result to history
                        history.append({"role": "assistant", "content": llm_response})
                        history.append({
                            "role": "user",
                            "content": f"Tool result: {tool_result}"
                        })
                        
                        turns_used += 1
                        continue  # Go back to top of loop for next turn
                    
                    except ValueError as e:
                        error_msg = f"Tool error: {str(e)}"
                        logger.error(f"[{session_id}] {error_msg}")
                        history.append({"role": "assistant", "content": error_msg})
                        final_response = error_msg
                        break
                    
                    except Exception as e:
                        error_msg = f"Tool execution error: {str(e)}"
                        logger.error(f"[{session_id}] {error_msg}")
                        history.append({"role": "assistant", "content": error_msg})
                        final_response = error_msg
                        break
                else:
                    logger.info(f"[{session_id}] Turn {turns_used}: NO JSON tool call detected - treating as plain text")
                    tool_call_detected = False
            
            except Exception as e:
                # Any error in parsing, treat as plain text
                logger.info(f"[{session_id}] Turn {turns_used}: Exception during tool extraction: {e} - treating as plain text")
                tool_call_detected = False
            
            # If not a tool call, this is the final answer
            if not tool_call_detected:
                history.append({"role": "assistant", "content": llm_response})
                final_response = llm_response
                logger.info(f"[{session_id}] Final response: plain text answer")
                break
        
        except requests.exceptions.Timeout:
            error_msg = "Model server timeout. Please try again."
            logger.error(f"[{session_id}] {error_msg}")
            history.append({"role": "assistant", "content": error_msg})
            final_response = error_msg
            break
        
        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to model server. Is it running?"
            logger.error(f"[{session_id}] {error_msg}")
            history.append({"role": "assistant", "content": error_msg})
            final_response = error_msg
            break
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"[{session_id}] {error_msg}")
            history.append({"role": "assistant", "content": error_msg})
            final_response = error_msg
            break
    
    # Check turn limit
    if turns_used > max_turns:
        final_response = "I've reached my turn limit. Please rephrase your question or break it into smaller parts."
        logger.warning(f"[{session_id}] Turn limit exceeded: {turns_used}")
    
    latency_ms = (time.time() - start_time) * 1000
    logger.info(f"[{session_id}] Completed in {latency_ms:.1f}ms, turns={turns_used}, tools={len(tool_calls_made)}")
    
    return {
        "response": final_response,
        "tool_calls_made": tool_calls_made,
        "turns_used": turns_used,
        "session_id": session_id,
        "latency_ms": round(latency_ms, 2)
    }


def extract_json_tool_call(text: str) -> Optional[Dict]:
    """Extract JSON tool call from text that may contain mixed content.
    
    Handles cases where LLM returns JSON followed by plain text explanation.
    Uses brace counting to properly extract nested JSON objects.
    
    Args:
        text: Response text that may contain JSON + plain text
        
    Returns:
        Parsed JSON dict if found, None otherwise
    """
    if '"tool"' not in text:
        return None
    
    # Find the first { before "tool"
    tool_pos = text.find('"tool"')
    start_pos = text.rfind('{', 0, tool_pos)
    
    if start_pos == -1:
        return None
    
    # Use brace counting to find matching closing brace
    brace_count = 0
    end_pos = start_pos
    
    for i in range(start_pos, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_pos = i + 1
                break
    
    try:
        json_str = text[start_pos:end_pos]
        parsed = json.loads(json_str)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def format_history_for_prompt(history: List[Dict[str, str]]) -> str:
    """Format message history into a prompt for the model.
    
    Args:
        history: List of messages with "role" and "content"
        
    Returns:
        Formatted prompt string
    """
    prompt = ""
    for msg in history:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        prompt += f"{role}: {content}\n\n"
    
    prompt += f"ASSISTANT: "
    return prompt
