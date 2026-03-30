#!/usr/bin/env python3
"""
Test script to generate grounded answers from the RAG pipeline.
Demonstrates end-to-end flow: retrieval -> context formatting -> LLM generation.

PREREQUISITES:
1. Model server must be running: cd model_server && python main.py
2. Index must be built: python index_documents.py (already done)
3. Requirements installed: pip install -r requirements.txt
"""

import sys
import json
from pipeline import RAGPipeline

def test_grounded_answer(query: str, top_k: int = 3):
    """
    Test grounded answer generation with a single query.
    
    Args:
        query: The user question
        top_k: Number of chunks to retrieve
    """
    print(f"\n{'='*80}")
    print(f"GROUNDED ANSWER TEST")
    print(f"{'='*80}\n")
    
    print(f"Query: {query}\n")
    
    try:
        # Initialize pipeline and load index
        print("[1/3] Loading RAG pipeline and index...")
        pipeline = RAGPipeline()
        pipeline.load_index()
        print(f"✓ Loaded index with {len(pipeline.chunks)} chunks\n")
        
        # Retrieve relevant chunks
        print(f"[2/3] Retrieving top {top_k} chunks...")
        retrieved = pipeline.retrieve(query, top_k=top_k, rerank=True)
        print(f"✓ Retrieved {len(retrieved)} chunks\n")
        
        for i, chunk in enumerate(retrieved, 1):
            print(f"  Chunk {i}:")
            print(f"    Source: {chunk['source_file']}")
            print(f"    Index: {chunk['chunk_index']}")
            print(f"    Score: {chunk['similarity_score']:.4f}")
            print(f"    Text: {chunk['text'][:100]}...\n")
        
        # Generate grounded answer (with streaming to see tokens as they arrive)
        print(f"[3/3] Generating answer from model server (streaming)...")
        result = pipeline.answer_with_sources(query, top_k=top_k, stream=True)
        print(f"\n✓ Answer generated\n")
        
        # Display results
        print(f"{'='*80}")
        print(f"ANSWER:")
        print(f"{'='*80}")
        print(f"\n{result['answer']}\n")
        
        print(f"{'='*80}")
        print(f"SOURCES (Attribution):")
        print(f"{'='*80}")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. {source['source_file']} (Chunk {source['chunk_index']})")
            print(f"   Similarity Score: {source['similarity_score']:.4f}\n")
        
        return result
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}", file=sys.stderr)
        print(f"\nDEBUG INFO:", file=sys.stderr)
        print(f"  - Is model_server running? (Check: http://localhost:8000/health)", file=sys.stderr)
        print(f"  - Index file exists? (Check: RAG/faiss_index.bin)", file=sys.stderr)
        print(f"  - Metadata file exists? (Check: RAG/chunks_metadata.json)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Example test queries
    test_queries = [
        "What is the collision deductible in the auto insurance?",
        "What is covered under comprehensive auto insurance?",
        "What is the annual deductible for the health plan?"
    ]
    
    print("\n" + "="*80)
    print("GROUNDED ANSWER GENERATION - TEST SUITE")
    print("="*80)
    
    # Run first query as example
    query = test_queries[0]
    result = test_grounded_answer(query)
    
    # Optionally test more queries
    print("\n\n" + "="*80)
    print("Try other queries:")
    print("="*80)
    for i, q in enumerate(test_queries[1:], 1):
        print(f"{i}. {q}")
    
    print("\nTo test another query interactively, edit this script or use:")
    print("  python -c \"from pipeline import RAGPipeline; p = RAGPipeline(); p.load_index(); print(p.answer_with_sources('Your question here'))\"")
