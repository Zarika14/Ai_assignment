# Multi-Turn Conversational Agent (Part 3)

A hand-rolled (no LangChain) multi-turn conversational agent that integrates with the LLM server and RAG pipeline to provide intelligent insurance policy queries, premium calculations, and claim status updates.

## Architecture

### Core Components

**agent.py** - Core agent logic:
- **Tool Functions**: 3 custom tools for insurance domain
  - `search_policy(query)` - Query policy documents via RAG pipeline
  - `calculate_premium(coverage, risk_score)` - Calculate insurance premiums
  - `check_claim_status(claim_id)` - Look up claim status
- **Session Management**: In-memory session storage with message history
- **Agent Loop**: Hand-rolled agentic loop with turn limiting (max 6 turns)
- **JSON Tool Calling**: Parse LLM responses for structured tool calls

**server.py** - FastAPI HTTP API:
- `POST /agent/chat` - Send message and get response
- `GET /agent/sessions/{session_id}` - Get session history
- `DELETE /agent/sessions/{session_id}` - Clear session
- `GET /agent/health` - Health check

**test_agent.py** - Comprehensive test suite:
- 3-turn conversation test with all tools
- Error handling validation
- Session persistence verification

## Tool Specification

### 1. search_policy(query: str)

**Purpose**: Search insurance policy documents using RAG pipeline

**Implementation**:
- Dynamically imports RAG pipeline from `rag/pipeline.py`
- Retrieves top 3 chunks matching the query
- Returns formatted string with sources and relevance scores

**Example**:
```python
result = search_policy("What is the collision deductible?")
# Returns: "Based on policy documents:\n\n1. [policy_auto_comprehensive.txt] The..."
```

### 2. calculate_premium(coverage: float, risk_score: float)

**Purpose**: Calculate insurance premium based on coverage and risk

**Formula**: `premium = coverage * 0.02 * risk_score`

**Implementation**:
- Input validation (must be positive numbers)
- Calculates annual and monthly premiums
- Returns formatted breakdown with base rate explanation

**Example**:
```python
result = calculate_premium(200000, 1.2)
# Returns: "Annual Premium: $4,800.00 (Base: 2% * Risk: 1.2x) Monthly: $400.00"
```

### 3. check_claim_status(claim_id: str)

**Purpose**: Retrieve claim status and details

**Implementation**:
- Hardcoded claims database with 5 sample claims
- CLM-001: approved ($5,000)
- CLM-002: pending ($3,500)
- CLM-003: rejected ($8,000)
- CLM-004: in_progress ($12,000)
- CLM-005: paid ($2,500)

**Example**:
```python
result = check_claim_status("CLM-003")
# Returns: "Claim CLM-003: Status=rejected, Amount=$8,000, Date=2024-01-20, Reason: Insufficient documentation"
```

## JSON Tool Calling Format

LLM responses are parsed for JSON tool calls in the following format:

```json
{
  "tool": "search_policy|calculate_premium|check_claim_status",
  "args": {
    "query": "...",
    "coverage": 200000,
    "risk_score": 1.2,
    "claim_id": "CLM-001"
  }
}
```

**Example Response**:
```python
{
  "response": "...",
  "tool_calls_made": ["search_policy", "calculate_premium"],
  "turns_used": 3,
  "session_id": "user-123",
  "latency_ms": 2345.67
}
```

## Agent Loop (Hand-Rolled Implementation)

The `run_turn()` function implements the core agentic loop:

1. **User Message** - Add to session history (Turn 1)
2. **LLM Call** - Send full history + system prompt to model server
3. **Response Parsing** - Check if response contains JSON tool call
4. **Tool Execution** - Validate tool exists, execute with error handling
5. **Result Addition** - Append tool result to history
6. **Loop Decision** - Continue if turns_used < max_turns (6)
7. **Final Response** - Return non-JSON response as final answer

**Key Features**:
- **Turn Limiting**: Max 6 turns prevents infinite loops
- **JSON Fallback**: Non-JSON responses treated as final answers
- **Error Recovery**: Malformed JSON gracefully handled
- **Tool Validation**: Unknown tools return error message, break loop
- **Session Persistence**: Full history maintained across turns

## Session Management

### Functions

**create_session(session_id: str)**
- Initialize new session with empty message history

**get_session_history(session_id: str) -> List[Dict]**
- Retrieve full message history for session

**delete_session(session_id: str) -> bool**
- Clear session (remove from in-memory dict)

**clear_all_sessions()**
- Reset all sessions (for testing)

### Storage

Sessions are stored in in-memory dictionary:
```python
sessions = {
    "user-123": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
```

**Optional PostgreSQL**: Set `DATABASE_URL` environment variable to persist sessions to database (not implemented in base version)

## API Endpoints

### POST /agent/chat

**Request**:
```json
{
  "session_id": "user-123",
  "message": "What's the deductible on my auto policy?"
}
```

**Response**:
```json
{
  "response": "Based on your auto comprehensive policy...",
  "tool_calls_made": ["search_policy"],
  "turns_used": 2,
  "session_id": "user-123",
  "latency_ms": 1234.56
}
```

### GET /agent/sessions/{session_id}

**Response**:
```json
{
  "session_id": "user-123",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "turn_count": 4
}
```

### DELETE /agent/sessions/{session_id}

**Response**:
```json
{
  "session_id": "user-123",
  "deleted": true
}
```

### GET /agent/health

**Response**:
```json
{
  "status": "ok",
  "service": "insurance-agent",
  "version": "1.0"
}
```

## Structured Logging

All operations logged with JSON output:

```json
{
  "timestamp": "2024-01-25T14:30:45.123Z",
  "event": "agent_chat",
  "session_id": "user-123",
  "input_tokens": 15,
  "latency_ms": 2345.67,
  "turns_used": 3,
  "tools_called": 2,
  "tool_calls_made": ["search_policy", "calculate_premium"]
}
```

## Error Handling

### Connection Errors
- Returns error message if model server is unreachable
- Logs connection details for debugging

### JSON Parsing Errors
- Malformed JSON automatically treated as plain text response
- No exception thrown, graceful degradation

### Tool Validation Errors
- Unknown tool name: Returns error, breaks loop
- Missing required args: Returns error, breaks loop
- Invalid arg types: Returns error, breaks loop

### Timeout Errors
- 60-second timeout on model server calls
- Returns timeout error message

## Testing

Run the comprehensive test suite:

```bash
python test_agent.py
```

**Tests Include**:
1. 3-turn conversation (policy query → premium calc → claim status)
2. Error handling scenarios (invalid tool, malformed args, max turns)
3. Session persistence (multi-message history accumulation)

**Expected Output**:
```
================================================================================
MULTI-TURN AGENT TEST - 3 TURN CONVERSATION
================================================================================

[TURN 1] User asks about deductible on auto policy
...
[TURN 2] User asks to calculate premium
...
[TURN 3] User asks about claim status
...

TEST SUMMARY
...
Total Tools Called: 3
✓ Test completed successfully!
```

## Running the Agent Server

### Start the Server

```bash
cd agent/
python server.py
```

Server runs on `http://localhost:8001`

### Simple Test Request

```bash
curl -X POST http://localhost:8001/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-123",
    "message": "What is the deductible on the auto policy?"
  }'
```

## System Prompt

The agent uses a specific system prompt that instructs the LLM how to use tools:

```
You are a helpful insurance policy assistant. You have access to the following tools:

1. search_policy(query) - Search policy documents for information
2. calculate_premium(coverage, risk_score) - Calculate insurance premium
3. check_claim_status(claim_id) - Check the status of a claim

When you need to use a tool, respond with ONLY valid JSON in this format:
{
  "tool": "tool_name",
  "args": {
    "param1": value1,
    "param2": value2
  }
}

If you do not need tools, respond with plain text answering the user's question.
```

## Dependencies

- `fastapi` - HTTP framework
- `uvicorn` - ASGI server
- `requests` - HTTP client (for model server calls)
- `structlog` - Structured logging
- `sentence-transformers` - For RAG pipeline (chunking, embedding)
- `faiss-cpu` - Vector index for RAG

## Integration Notes

- **Model Server**: Requires `/chat` endpoint at `http://localhost:8000` (configurable)
- **RAG Pipeline**: Dynamically imported from `rag/pipeline.py` when search_policy tool is called
- **Policy Documents**: Loaded from `rag/documents/` directory

## Future Enhancements

1. **PostgreSQL Session Persistence** - Store sessions in database
2. **Tool Parameter Validation** - JSON schema validation for tool args
3. **Multi-LLM Support** - Route to different model servers per tool
4. **Rate Limiting** - Per-session rate limits
5. **Conversation Analytics** - Track tool usage patterns, success rates
6. **Fine-Tuning** - Optimize model for insurance domain
