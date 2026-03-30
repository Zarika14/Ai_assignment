#!/usr/bin/env python3
"""
Simple test script for model_server endpoints.
Run this when the FastAPI server is running: python main.py
Ollama must be running: ollama serve
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"
OLLAMA_URL = "http://localhost:11434"

def test_health():
    """Test the health check endpoint"""
    print("\n" + "="*60)
    print("TEST 1: GET /health")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model"] == "mistral:latest"
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_chat():
    """Test the non-streaming chat endpoint"""
    print("\n" + "="*60)
    print("TEST 2: POST /chat (non-streaming)")
    print("="*60)
    
    try:
        payload = {
            "message": "What is the capital of France?",
            "system_prompt": "You are a helpful assistant."
        }
        
        print(f"Request Body: {json.dumps(payload, indent=2)}")
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response:")
        print(f"  - Model: {result.get('model')}")
        print(f"  - Tokens Used: {result.get('tokens_used')}")
        print(f"  - Response: {result.get('response')[:100]}..." if len(result.get('response', '')) > 100 else f"  - Response: {result.get('response')}")
        
        assert response.status_code == 200
        assert result.get("model") == "mistral:latest"
        assert result.get("response")
        assert isinstance(result.get("tokens_used"), int)
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_chat_stream():
    """Test the streaming chat endpoint"""
    print("\n" + "="*60)
    print("TEST 3: POST /chat/stream (SSE streaming)")
    print("="*60)
    
    try:
        payload = {
            "message": "Write a short haiku about AI",
            "system_prompt": "You are a poet."
        }
        
        print(f"Request Body: {json.dumps(payload, indent=2)}")
        print(f"Streaming response (tokens as they arrive):")
        print("-" * 40)
        
        response = requests.post(
            f"{BASE_URL}/chat/stream",
            json=payload,
            stream=True
        )
        
        print(f"Status Code: {response.status_code}")
        
        token_count = 0
        full_response = ""
        
        for line in response.iter_lines():
            if not line:
                # Skip empty lines
                continue
            
            line = line.decode('utf-8') if isinstance(line, bytes) else line
            
            # Check if this is an SSE data line
            if not line.startswith("data: "):
                # Skip non-SSE lines (comments, etc.)
                continue
            
            try:
                # Parse SSE event - remove "data: " prefix
                event_json = line[6:].strip()
                if not event_json:
                    continue
                
                event_data = json.loads(event_json)
                token = event_data.get("token", "")
                done = event_data.get("done", False)
                error = event_data.get("error", None)
                
                if error:
                    print(f"\n❌ Error from server: {error}")
                    raise Exception(f"Server error: {error}")
                
                if token:
                    print(token, end="", flush=True)
                    full_response += token
                    token_count += 1
                
                if done:
                    print("\n" + "-" * 40)
                    break
            except json.JSONDecodeError as e:
                print(f"\n⚠️  Failed to parse SSE event: {repr(line)}")
                print(f"   Error: {e}")
                continue
        
        print(f"Tokens received: {token_count}")
        print(f"Full response: {full_response[:100]}..." if len(full_response) > 100 else f"Full response: {full_response}")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert token_count > 0, f"Expected at least 1 token, got {token_count}"
        assert full_response, "Expected non-empty response"
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_ollama():
    """Check if Ollama is running"""
    print("\n" + "="*60)
    print("PRE-CHECK: Ollama Status")
    print("="*60)
    
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        models = response.json().get("models", [])
        model_names = [m.get("name") for m in models]
        print(f"Ollama is running with models: {model_names}")
        
        if "mistral:latest" in model_names:
            print("✓ mistral:latest is available")
            return True
        else:
            print("✗ mistral:latest not found. Available models:")
            for name in model_names:
                print(f"  - {name}")
            return False
    except Exception as e:
        print(f"✗ FAILED to connect to Ollama: {e}")
        print(f"  Make sure Ollama is running: ollama serve")
        return False


def check_server():
    """Check if FastAPI server is running"""
    print("\n" + "="*60)
    print("PRE-CHECK: FastAPI Server Status")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✓ Server is running and responds to /health")
        return True
    except Exception as e:
        print(f"✗ FAILED to connect to server: {e}")
        print(f"  Make sure the server is running: python main.py")
        return False


def main():
    print("\n" + "🚀 "*20)
    print("Model Server Test Suite")
    print("🚀 "*20)
    
    # Pre-checks
    ollama_ok = check_ollama()
    server_ok = check_server()
    
    if not ollama_ok or not server_ok:
        print("\n❌ Pre-checks failed. Exiting.")
        sys.exit(1)
    
    # Run tests
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Non-Streaming Chat", test_chat()))
    results.append(("Streaming Chat", test_chat_stream()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\nResults: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
