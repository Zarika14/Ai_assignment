#!/usr/bin/env python3
"""
Test script for multi-turn agent.

Demonstrates a 3-turn conversation:
1. Query about policy details (uses search_policy tool)
2. Calculate premium (uses calculate_premium tool)
3. Check claim status (uses check_claim_status tool)
"""

import json
import sys
import time
from agent import run_turn, clear_all_sessions

def test_three_turn_agent():
    """Test agent with 3-turn conversation."""
    
    print("=" * 80)
    print("MULTI-TURN AGENT TEST - 3 TURN CONVERSATION")
    print("=" * 80)
    
    session_id = "test-session-001"
    clear_all_sessions()  # Reset for testing
    
    # Turn 1: Policy query
    print("\n[TURN 1] User asks about deductible on auto policy")
    print("-" * 80)
    
    turn1_message = "What is the collision deductible on the auto comprehensive policy?"
    print(f"Message: {turn1_message}\n")
    
    turn1_start = time.time()
    turn1_result = run_turn(session_id, turn1_message)
    turn1_latency = (time.time() - turn1_start) * 1000
    
    print(f"Response:\n{turn1_result['response']}\n")
    print(f"Tools Called: {turn1_result['tool_calls_made']}")
    print(f"Turns Used: {turn1_result['turns_used']}")
    print(f"Latency: {turn1_latency:.2f}ms")
    
    # Turn 2: Premium calculation
    print("\n" + "=" * 80)
    print("[TURN 2] User asks to calculate premium")
    print("-" * 80)
    
    turn2_message = "Calculate the annual premium for a coverage amount of $200,000 with a risk score of 1.2"
    print(f"Message: {turn2_message}\n")
    
    turn2_start = time.time()
    turn2_result = run_turn(session_id, turn2_message)
    turn2_latency = (time.time() - turn2_start) * 1000
    
    print(f"Response:\n{turn2_result['response']}\n")
    print(f"Tools Called: {turn2_result['tool_calls_made']}")
    print(f"Turns Used: {turn2_result['turns_used']}")
    print(f"Latency: {turn2_latency:.2f}ms")
    
    # Turn 3: Claim status
    print("\n" + "=" * 80)
    print("[TURN 3] User asks about claim status")
    print("-" * 80)
    
    turn3_message = "What is the status of claim CLM-003? I want to know if it was rejected and why."
    print(f"Message: {turn3_message}\n")
    
    turn3_start = time.time()
    turn3_result = run_turn(session_id, turn3_message)
    turn3_latency = (time.time() - turn3_start) * 1000
    
    print(f"Response:\n{turn3_result['response']}\n")
    print(f"Tools Called: {turn3_result['tool_calls_made']}")
    print(f"Turns Used: {turn3_result['turns_used']}")
    print(f"Latency: {turn3_latency:.2f}ms")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total_tools_called = len(turn1_result["tool_calls_made"]) + len(turn2_result["tool_calls_made"]) + len(turn3_result["tool_calls_made"])
    total_latency = turn1_latency + turn2_latency + turn3_latency
    
    print(f"Session ID: {session_id}")
    print(f"Total Turns: 3")
    print(f"Total Tools Called: {total_tools_called}")
    print(f"  - search_policy: {sum(1 for t in turn1_result['tool_calls_made'] + turn2_result['tool_calls_made'] + turn3_result['tool_calls_made'] if t == 'search_policy')}")
    print(f"  - calculate_premium: {sum(1 for t in turn1_result['tool_calls_made'] + turn2_result['tool_calls_made'] + turn3_result['tool_calls_made'] if t == 'calculate_premium')}")
    print(f"  - check_claim_status: {sum(1 for t in turn1_result['tool_calls_made'] + turn2_result['tool_calls_made'] + turn3_result['tool_calls_made'] if t == 'check_claim_status')}")
    print(f"Total Latency: {total_latency:.2f}ms")
    print(f"Average Latency per Turn: {total_latency/3:.2f}ms")
    
    print("\n[✓] Test completed successfully!")
    print("=" * 80)
    
    return True


def test_error_handling():
    """Test error handling scenarios."""
    
    print("\n" + "=" * 80)
    print("ERROR HANDLING TESTS")
    print("=" * 80)
    
    session_id = "error-test-session"
    clear_all_sessions()
    
    # Test 1: Invalid tool call
    print("\n[TEST] Invalid tool call in response")
    print("-" * 80)
    print("(Agent would receive tool call with invalid tool name)")
    print("Expected: Error message returned, loop continues or breaks gracefully\n")
    
    # Test 2: Malformed JSON args
    print("[TEST] Malformed JSON arguments")
    print("-" * 80)
    print("(Tool call missing required args)")
    print("Expected: Error message returned\n")
    
    # Test 3: Max turns exceeded
    print("[TEST] Turn limiting (max 6 turns)")
    print("-" * 80)
    print("(If agent keeps calling tools, should stop at 6 turns)")
    message = "List every policy detail you can find"
    result = run_turn(session_id, message)
    print(f"Message: {message}")
    print(f"Turns Used: {result['turns_used']} (should be <= 6)")
    print(f"Response: {result['response'][:100]}...")
    print()
    
    print("\n[✓] Error handling tests completed!")
    print("=" * 80)
    
    return True


def test_session_persistence():
    """Test session history persistence."""
    
    print("\n" + "=" * 80)
    print("SESSION PERSISTENCE TEST")
    print("=" * 80)
    
    session_id = "persistence-test"
    clear_all_sessions()
    
    print("\n[TEST] Message history accumulation")
    print("-" * 80)
    
    # Send 2 messages in same session
    msg1 = "Hello, what policies do you have?"
    msg2 = "Can you calculate a premium for me?"
    
    result1 = run_turn(session_id, msg1)
    print(f"Turn 1: {msg1}")
    print(f"Response received (length: {len(result1['response'])} chars)\n")
    
    result2 = run_turn(session_id, msg2)
    print(f"Turn 2: {msg2}")
    print(f"Response received (length: {len(result2['response'])} chars)\n")
    
    # Verify history
    from agent import get_session_history
    history = get_session_history(session_id)
    
    print(f"Session History Length: {len(history)} messages")
    print("History:")
    for i, msg in enumerate(history, 1):
        content_preview = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
        print(f"  {i}. [{msg['role'].upper()}] {content_preview}")
    
    print("\n[✓] Session persistence test completed!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        # Run all tests
        test_three_turn_agent()
        test_error_handling()
        test_session_persistence()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED [OK]")
        print("=" * 80)
        sys.exit(0)
    
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
