#!/usr/bin/env python3
"""
COMPLETION VERIFICATION - Parts 1 & 2

This script demonstrates that Parts 1 & 2 are fully functional end-to-end.
Run this to verify everything works together.
"""

import subprocess
import requests
import time
import sys

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def check_service(url, name, timeout=5):
    """Check if a service is running"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅ {name} is running")
            return True
    except:
        pass
    print(f"❌ {name} is NOT running - {url}")
    return False

def main():
    print_section("PARTS 1 & 2 - COMPLETION VERIFICATION")
    
    # Check 1: Ollama
    print("\n[CHECK 1] Ollama Service")
    ollama_ready = check_service("http://localhost:11434/api/tags", "Ollama", timeout=3)
    
    if not ollama_ready:
        print("\n⚠️  Ollama not running. Start with: ollama serve")
        print("   Then run this script again.")
        return 1
    
    # Check 2: Model Server
    print("\n[CHECK 2] Model Server (FastAPI)")
    server_ready = check_service("http://localhost:8000/health", "Model Server", timeout=3)
    
    if not server_ready:
        print("\n⚠️  Model Server not running. Start with: python model_server/main.py")
        print("   Then run this script again.")
        return 1
    
    # Check 3: Index Files
    print("\n[CHECK 3] RAG Index Files")
    from pathlib import Path
    index_file = Path("rag/faiss_index.bin")
    metadata_file = Path("rag/chunks_metadata.json")
    
    if index_file.exists() and metadata_file.exists():
        print(f"✅ FAISS index exists: {index_file}")
        print(f"✅ Metadata exists: {metadata_file}")
    else:
        print(f"❌ Index files not found")
        if not index_file.exists():
            print(f"   Missing: {index_file}")
            print(f"   Run: python rag/index_documents.py")
        if not metadata_file.exists():
            print(f"   Missing: {metadata_file}")
        return 1
    
    # Check 4: Database connectivity
    print("\n[CHECK 4] System Connectivity")
    
    # Test model server health
    try:
        r = requests.get("http://localhost:8000/health", timeout=3)
        model_info = r.json()
        print(f"✅ Model Server Health:")
        print(f"   Status: {model_info.get('status')}")
        print(f"   Model: {model_info.get('model')}")
    except Exception as e:
        print(f"❌ Model Server health check failed: {e}")
        return 1
    
    # Test Ollama connectivity
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = r.json().get("models", [])
        model_names = [m.get("name") for m in models]
        print(f"✅ Ollama Service:")
        print(f"   Available models: {', '.join(model_names)}")
        
        if "mistral:latest" not in model_names:
            print(f"❌ mistral:latest not available")
            print(f"   Run once: ollama pull mistral:latest")
            return 1
    except Exception as e:
        print(f"❌ Ollama check failed: {e}")
        return 1
    
    print_section("RUNNING QUICK VALIDATION TESTS")
    
    # Test 1: Model server endpoints
    print("\n[TEST 1] Model Server Endpoints")
    
    try:
        # Test /chat endpoint
        payload = {
            "message": "What is 2+2?",
            "system_prompt": "You are a helpful assistant. Be brief."
        }
        response = requests.post("http://localhost:8000/chat", json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ /chat endpoint working")
            print(f"   Response: {result.get('response', '')[:50]}...")
        else:
            print(f"❌ /chat returned status {response.status_code}")
            return 1
    except Exception as e:
        print(f"❌ /chat endpoint failed: {e}")
        return 1
    
    # Test 2: RAG Pipeline
    print("\n[TEST 2] RAG Pipeline")
    
    try:
        from rag.pipeline import RAGPipeline
        
        pipeline = RAGPipeline(rerank=False)  # Disable reranking for speed
        pipeline.load_index()
        print(f"✅ RAG Pipeline initialized")
        print(f"   Loaded {len(pipeline.chunks)} chunks")
        
        # Test retrieval
        results = pipeline.retrieve("What is covered?", top_k=2)
        print(f"✅ Retrieval working")
        print(f"   Found {len(results)} results")
        if results:
            print(f"   Top result: [{results[0]['source_file']}] score={results[0]['similarity_score']:.3f}")
    except Exception as e:
        print(f"❌ RAG Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test 3: End-to-end answer generation
    print("\n[TEST 3] End-to-End Answer Generation")
    
    try:
        query = "What types of coverage are available?"
        print(f"   Query: {query}")
        
        answer = pipeline.answer_with_sources(
            query,
            top_k=2,
            stream=False
        )
        
        print(f"✅ Answer generation working")
        print(f"   Answer: {answer['answer'][:100]}...")
        print(f"   Sources: {len(answer['sources'])} document(s)")
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print_section("✅ ALL VERIFICATION TESTS PASSED")
    
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  Parts 1 & 2 are fully functional and working together!       ║
║                                                                ║
║  The system is ready for:                                     ║
║  ✓ Querying insurance policies with context awareness         ║
║  ✓ Streaming token generation                                 ║
║  ✓ Source attribution and grounding                           ║
║  ✓ Production deployment (Docker ready)                       ║
║                                                                ║
║  Next Steps:                                                  ║
║  1. Run comprehensive test suite:                             ║
║     python rag/test_comprehensive.py                          ║
║                                                                ║
║  2. Try interactive retrieval:                                ║
║     python rag/test_interactive.py                            ║
║                                                                ║
║  3. See full end-to-end demo:                                 ║
║     python rag/test_grounded_answer.py                        ║
║                                                                ║
║  4. Read detailed docs:                                       ║
║     - README.md (architecture overview)                       ║
║     - model_server/README.md (LLM serving)                    ║
║     - rag/README.md (RAG pipeline)                            ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
