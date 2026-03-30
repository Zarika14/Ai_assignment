# Part 3 Logging & Tool Detection Fixes

## Problem Identified
The agent test showed `Tools Called: 0` even though the LLM was returning tool calls in JSON format. The tools were not being executed.

**Original Output**:
```
Response:
 {"tool": "search_policy", "args": {"query": "collision deductible..."}}
The collision deductible for... is $500.

Tools Called: []  ← SHOULD BE ['search_policy']
Turns Used: 1
```

## Root Cause Analysis

### Issue 1: JSON Extraction Failed
The LLM response had **mixed content** (JSON + plain text explanation):
```
{"tool": "search_policy", "args": {...}}  ← JSON tool call
The collision deductible for... is $500.   ← Plain text explanation
```

The original regex pattern `\{[^{}]*"tool"[^{}]*\}` **failed** because:
- It doesn't allow ANY braces inside the pattern
- The `"args"` object contains nested braces: `{"query": "..."}`
- Pattern couldn't match: `{"tool": "search_policy", "args": {"query": "..."}}`

### Issue 2: Logging Was Silent
- Tool extraction failures showed no logs
- Tools weren't being counted in results
- Hard to debug what was happening

### Issue 3: RAG Import Error
- Path handling for RAG pipeline was incorrect
- `sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))` added the `rag` folder, not the parent
- Should add the parent (`Ai_assignment`) folder to properly import `rag.pipeline`

## Solutions Implemented

### Solution 1: Smart JSON Extraction Function
Created `extract_json_tool_call()` that:
- Uses **brace counting** to find matching closing brace
- Handles nested JSON objects properly
- Works with mixed content (JSON + text)

**How it works**:
```python
def extract_json_tool_call(text: str) -> Optional[Dict]:
    # Find position of "tool" key
    tool_pos = text.find('"tool"')
    start_pos = text.rfind('{', 0, tool_pos)  # Find opening brace before it
    
    # Count braces to find matching close
    brace_count = 0
    for i in range(start_pos, len(text)):
        if text[i] == '{': brace_count += 1
        elif text[i] == '}': brace_count -= 1
        if brace_count == 0: 
            end_pos = i + 1
            break
    
    # Parse JSON from start to end position
    json_str = text[start_pos:end_pos]
    return json.loads(json_str)
```

**Example**:
```
Input: {"tool": "search_policy", "args": {"query": "deductible"}}...extra text...
Output: {"tool": "search_policy", "args": {"query": "deductible"}}
```

### Solution 2: Enhanced Logging
Added detailed logging at each step:

**Before**:
```
[test-session-001] Tool executed: search_policy
```

**After**:
```
[test-session-001] Turn 1: JSON tool call extracted successfully
[test-session-001] Turn 1: Tool call DETECTED: search_policy with args {'query': '...'}
[test-session-001] Turn 1: Tool EXECUTED successfully: search_policy, result length=240
```

Logs now show:
- When JSON extraction succeeds/fails
- Tool name and arguments being used
- Result length for verification
- Turn number for debugging multi-turn flow

### Solution 3: Fixed RAG Path Handling
**Before**:
```python
sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))  # Wrong!
from rag.pipeline import RAGPipeline
```
This added `/Ai_assignment/rag` to path, but then tries to import `rag.pipeline` (module not found)

**After**:
```python
parent_dir = Path(__file__).parent.parent  # /Ai_assignment/
sys.path.insert(0, str(parent_dir))        # Add /Ai_assignment/
from rag.pipeline import RAGPipeline       # Now finds rag/ folder inside
```

## Logging Output Comparison

### Before (Broken):
```
[test-session-001] Turn 1: NO JSON pattern found in response - treating as plain text
[test-session-001] Final response: plain text answer

Tools Called: []  ← Wrong!
```

### After (Fixed):
```
[test-session-001] Turn 1: JSON tool call extracted successfully
[test-session-001] Turn 1: Tool call DETECTED: search_policy with args {'query': 'collision deductible...'}
[test-session-001] Turn 1: Tool EXECUTED successfully: search_policy, result length=240
[test-session-001] Turn 2: Calling model server (continuing agentic loop)

Tools Called: ['search_policy']  ← Correct!
```

## Test Results

After fixes, the test now shows:

```
[TURN 1] User asks about deductible on auto policy
Message: What is the collision deductible on the auto comprehensive policy?
Tools Called: ['search_policy']  ✓ Tool was executed!
Turns Used: 2  ✓ Loop continued to get final answer
```

### Log Flow Example

**Turn-by-turn execution**:
$$\text{Turn 1: User message} \rightarrow \text{Turn 2: LLM calls search_policy} \rightarrow \text{Turn 3: LLM reads tool result} \rightarrow \text{Turn 4: LLM gives final answer} \rightarrow \text{STOP}$$

**Logs show**:
```
Turn 1: Calling model server
Turn 1: Got response (length=167)
Turn 1: JSON tool call extracted successfully          ← JSON found!
Turn 1: Tool call DETECTED: search_policy             ← Tool identified!
Turn 1: Tool EXECUTED successfully: search_policy     ← Executed!
Turn 2: Calling model server                          ← Loop continued
Turn 2: Got response (length=343)
Turn 2: NO JSON tool call detected - treating as plain text  ← Final answer
```

## What We Fixed

| Issue | Solution | Result |
|-------|----------|--------|
| JSON not extracted from mixed content | Implemented brace-counting extraction | Tools now properly detected |
| Unknown tool in response | LLM tried calling non-existent tools | Validate tool exists before executing |
| Silent failures | Added logging at each step | Detailed trace of what happened |
| RAG import failed | Fixed sys.path handling | RAG pipeline can be imported |
| Unicode errors in tests | Replaced checkmarks with ASCII | Tests run on Windows |

## Validation

Run the test to see the improvements:

```bash
python .\agent\test_agent.py
```

Expected output:
```
[TURN 1] User asks about deductible on auto policy
...
Tools Called: ['search_policy']
Turns Used: 2
✓ Test completed successfully!
```

## Key Improvements for Production

1. **Reliability**: Handles mixed JSON + text responses from LLMs
2. **Debuggability**: Full logging shows exactly what's happening
3. **Correctness**: tools_called list accurately reflects executed tools
4. **Integration**: RAG pipeline properly integrated with agent
5. **Cross-platform**: Works on Windows (fixed Unicode issues)
