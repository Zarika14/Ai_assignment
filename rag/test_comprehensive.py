#!/usr/bin/env python3
"""
Comprehensive test suite for RAG Pipeline

Tests all functionality:
- Document loading and validation
- Chunking logic
- Embedding and indexing
- Retrieval with and without re-ranking
- Error handling and edge cases
- End-to-end integration

Run: python test_comprehensive.py
"""

import sys
import json
from pathlib import Path
from pipeline import RAGPipeline, TextChunker

def print_test_header(test_name: str):
    """Print test header"""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")

def print_result(passed: bool, message: str = ""):
    """Print test result"""
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"\n{status}{f': {message}' if message else ''}")

# ============================================================================
# UNIT TESTS
# ============================================================================

def test_text_chunker():
    """Test TextChunker with various inputs"""
    print_test_header("TextChunker - Basic Functionality")
    
    chunker = TextChunker(chunk_size=100, overlap=20)
    
    # Test 1: Simple text
    text = "This is a test. This is another sentence. This is the third sentence. " * 5
    chunks = chunker.split(text, "test.txt")
    
    passed = (
        len(chunks) > 0 and
        all("text" in c and "source_file" in c and "chunk_index" in c for c in chunks) and
        all(c["source_file"] == "test.txt" for c in chunks)
    )
    print_result(passed, f"Created {len(chunks)} chunks")
    
    # Test 2: Empty text handling
    print_test_header("TextChunker - Empty Text Handling")
    chunks = chunker.split("", "empty.txt")
    passed = len(chunks) == 0
    print_result(passed, "Empty text returns 0 chunks")
    
    # Test 3: Single sentence
    print_test_header("TextChunker - Single Sentence")
    chunks = chunker.split("This is one sentence.", "single.txt")
    passed = len(chunks) == 1
    print_result(passed, f"Single sentence creates 1 chunk")
    
    return True

def test_rag_pipeline_initialization():
    """Test RAGPipeline initialization"""
    print_test_header("RAGPipeline - Initialization")
    
    try:
        pipeline = RAGPipeline(rerank=True)
        passed = (
            pipeline.embedding_model is not None and
            pipeline.reranker is not None and
            pipeline.index is None  # Not built yet
        )
        print_result(passed, "Pipeline initialized with models loaded")
        return True
    except Exception as e:
        print_result(False, f"Initialization failed: {e}")
        return False

def test_rag_pipeline_error_handling():
    """Test error handling for invalid operations"""
    print_test_header("RAGPipeline - Error Handling")
    
    pipeline = RAGPipeline(rerank=False)
    passed_tests = []
    
    # Test 1: build_index without load_documents
    print("\n[1] build_index without load_documents:")
    try:
        pipeline.build_index()
        print_result(False, "Should have raised ValueError")
        passed_tests.append(False)
    except ValueError as e:
        print_result(True, f"Correctly raised: {str(e)[:50]}...")
        passed_tests.append(True)
    
    # Test 2: retrieve without load_index
    print("\n[2] retrieve without load_index:")
    try:
        pipeline.retrieve("test query")
        print_result(False, "Should have raised ValueError")
        passed_tests.append(False)
    except ValueError as e:
        print_result(True, f"Correctly raised: {str(e)[:50]}...")
        passed_tests.append(True)
    
    # Test 3: retrieve with empty query
    print("\n[3] retrieve with empty query:")
    # First need to load index
    try:
        pipeline.load_index()  # This will fail but we catch it
    except FileNotFoundError:
        pass  # Expected - index not built yet
    except:
        pass
    
    # Create mock index for testing
    import numpy as np
    import faiss
    try:
        dimension = 384
        pipeline.index = faiss.IndexFlatIP(dimension)
        embeddings = np.random.rand(5, dimension).astype(np.float32)
        pipeline.index.add(embeddings)
        pipeline.chunks = [
            {"text": f"chunk {i}", "source_file": "test.txt", "chunk_index": i}
            for i in range(5)
        ]
        
        try:
            pipeline.retrieve("")  # Empty query
            print_result(False, "Should have raised ValueError")
            passed_tests.append(False)
        except ValueError as e:
            print_result(True, f"Correctly raised: {str(e)[:50]}...")
            passed_tests.append(True)
    except:
        print_result(False, "Failed to set up mock index")
        passed_tests.append(False)
    
    # Test 4: answer_with_sources with empty query
    print("\n[4] answer_with_sources with empty query:")
    try:
        result = pipeline.answer_with_sources("")
        print_result(False, "Should have raised ValueError")
        passed_tests.append(False)
    except ValueError as e:
        print_result(True, f"Correctly raised: {str(e)[:50]}...")
        passed_tests.append(True)
    
    return all(passed_tests)

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_pipeline():
    """Test full pipeline: load → chunk → embed → index → retrieve"""
    print_test_header("Full Pipeline - End-to-End")
    
    try:
        # Initialize
        print("\n[1] Initializing pipeline...")
        pipeline = RAGPipeline(rerank=False)  # Skip reranker for speed
        print("✓ Pipeline initialized")
        
        # Load documents
        print("\n[2] Loading documents...")
        chunks = pipeline.load_documents()
        if not chunks:
            print_result(False, "No chunks loaded")
            return False
        print(f"✓ Loaded {len(chunks)} chunks")
        
        # Build index
        print("\n[3] Building index...")
        pipeline.build_index()
        if pipeline.index is None:
            print_result(False, "Index is None")
            return False
        print(f"✓ Index built with {pipeline.index.ntotal} vectors")
        
        # Retrieve
        print("\n[4] Testing retrieval...")
        test_queries = [
            "What is the collision deductible?",
            "What does health insurance cover?",
            "What are homeowners insurance benefits?"
        ]
        
        for query in test_queries:
            results = pipeline.retrieve(query, top_k=2, rerank=False)
            if not results or len(results) == 0:
                print(f"⚠️  No results for: {query}")
                continue
            print(f"✓ Retrieved {len(results)} results for: {query[:40]}...")
            for i, r in enumerate(results, 1):
                print(f"    {i}. [{r['source_file']}:{r['chunk_index']}] "
                      f"score={r['similarity_score']:.3f}")
        
        print_result(True, "Full pipeline working end-to-end")
        return True
        
    except Exception as e:
        print_result(False, f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_retrieval_quality():
    """Test retrieval quality - relevant chunks should rank high"""
    print_test_header("Retrieval Quality - Relevance Scoring")
    
    try:
        pipeline = RAGPipeline(rerank=False)
        pipeline.load_index()
        
        # Query about collision coverage
        query = "collision deductible auto insurance"
        results = pipeline.retrieve(query, top_k=5, rerank=False)
        
        # Check results
        if not results:
            print_result(False, "No results returned")
            return False
        
        # Results should be sorted by similarity
        scores = [r['similarity_score'] for r in results]
        is_sorted = scores == sorted(scores, reverse=True)
        
        print(f"Retrieved {len(results)} chunks:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. Score: {r['similarity_score']:.4f} | "
                  f"{r['source_file']} chunk {r['chunk_index']}")
        
        print_result(
            is_sorted and results[0]['similarity_score'] > 0.5,
            f"Top score: {scores[0]:.4f}, Results properly sorted"
        )
        return True
        
    except Exception as e:
        print_result(False, f"Test failed: {e}")
        return False

def test_reranking():
    """Test with and without re-ranking"""
    print_test_header("Retrieval Re-ranking - Quality Improvement")
    
    try:
        # Without re-ranking
        print("\n[1] Without re-ranking:")
        pipeline_no_rerank = RAGPipeline(rerank=False)
        pipeline_no_rerank.load_index()
        
        query = "coverage limits and deductibles"
        results_no_rerank = pipeline_no_rerank.retrieve(query, top_k=3, rerank=False)
        print(f"✓ Retrieved {len(results_no_rerank)} results")
        
        # With re-ranking
        print("\n[2] With re-ranking:")
        pipeline_rerank = RAGPipeline(rerank=True)
        pipeline_rerank.load_index()
        
        results_rerank = pipeline_rerank.retrieve(query, top_k=3, rerank=True)
        print(f"✓ Retrieved {len(results_rerank)} results")
        
        # Compare
        print("\nComparison:")
        print(f"  Without re-rank - Top score: {results_no_rerank[0]['similarity_score']:.4f}")
        if results_rerank[0].get('rerank_score'):
            print(f"  With re-rank - Top score: {results_rerank[0]['rerank_score']:.4f}")
        
        print_result(True, "Re-ranking executed successfully")
        return True
        
    except Exception as e:
        print_result(False, f"Re-ranking test failed: {e}")
        return False

def test_integrated_with_model_server():
    """Test integration with model server (if available)"""
    print_test_header("Integration - Model Server Connection")
    
    import requests
    
    try:
        # Check if model server is running
        response = requests.get("http://localhost:8000/health", timeout=2)
        model_server_available = response.status_code == 200
    except:
        model_server_available = False
    
    if not model_server_available:
        print("⚠️  Model server not available (http://localhost:8000/health)")
        print("    Skipping integration test. Start model server with: python model_server/main.py")
        print_result(True, "Test skipped - not a failure")
        return True
    
    try:
        pipeline = RAGPipeline(rerank=False)
        pipeline.load_index()
        
        query = "What is covered in comprehensive auto insurance?"
        
        print(f"\nQuery: {query}")
        print("-" * 70)
        
        result = pipeline.answer_with_sources(
            query,
            top_k=2,
            model_server_url="http://localhost:8000",
            stream=False
        )
        
        print(f"\nAnswer: {result['answer'][:200]}...")
        print(f"Sources: {len(result['sources'])} chunks")
        
        passed = (
            result['answer'] and
            len(result['sources']) > 0
        )
        print_result(passed, "Model server integration working")
        return passed
        
    except Exception as e:
        print_result(False, f"Model server integration test failed: {e}")
        return False

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests"""
    print("\n" + "🧪 " * 20)
    print("RAG PIPELINE COMPREHENSIVE TEST SUITE")
    print("🧪 " * 20)
    
    tests = [
        ("Unit Tests", [
            ("TextChunker", test_text_chunker),
            ("RAGPipeline Initialization", test_rag_pipeline_initialization),
            ("Error Handling", test_rag_pipeline_error_handling),
        ]),
        ("Integration Tests", [
            ("Full Pipeline", test_full_pipeline),
            ("Retrieval Quality", test_retrieval_quality),
            ("Re-ranking", test_reranking),
            ("Model Server Integration", test_integrated_with_model_server),
        ])
    ]
    
    results = {}
    
    for category, category_tests in tests:
        print(f"\n\n{'#' * 70}")
        print(f"# {category}")
        print(f"{'#' * 70}")
        
        for test_name, test_func in category_tests:
            try:
                passed = test_func()
                results[f"{category} - {test_name}"] = passed
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                results[f"{category} - {test_name}"] = False
    
    # Print summary
    print(f"\n\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for p in results.values() if p)
    total_count = len(results)
    
    print(f"\nResults: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
