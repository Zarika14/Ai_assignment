#!/usr/bin/env python3
"""
Interactive RAG Retrieval Test Script

Type your questions and see which policy chunks are retrieved.
Type 'exit' or 'quit' to stop.
"""

from pipeline import RAGPipeline

def main():
    """Run interactive retrieval test."""
    print("\n" + "="*70)
    print("INTERACTIVE RAG RETRIEVAL TEST")
    print("="*70)
    print("\nLoading RAG pipeline...")
    
    try:
        pipeline = RAGPipeline(rerank=True)
        pipeline.load_index()
        print("✓ Index loaded\n")
    except FileNotFoundError:
        print("❌ ERROR: Index not found!")
        print("   Run: python index_documents.py")
        return
    
    print("Type your questions about insurance policies.")
    print("Examples:")
    print("  - What is the collision deductible?")
    print("  - What does homeowners insurance cover?")
    print("  - What are the copay amounts?")
    print("  - Is flood damage covered?")
    print("\nType 'exit' or 'quit' to stop.\n")
    print("-"*70)
    
    while True:
        try:
            # Get user query
            query = input("\n📋 Your question: ").strip()
            
            if not query:
                print("⚠️  Please enter a question")
                continue
            
            if query.lower() in ['exit', 'quit']:
                print("\n✓ Goodbye!")
                break
            
            # Retrieve results
            print(f"\n🔍 Retrieving relevant chunks...\n")
            results = pipeline.retrieve(query, top_k=3, rerank=True)
            
            if not results:
                print("❌ No relevant chunks found")
                continue
            
            # Display results
            print("="*70)
            print(f"RETRIEVED CHUNKS ({len(results)}):")
            print("="*70)
            
            for i, result in enumerate(results, 1):
                score = result['similarity_score']
                if 'rerank_score' in result:
                    print(f"\n{i}. 📄 {result['source_file']} (chunk {result['chunk_index']})")
                    print(f"   Similarity Score: {score:.3f}")
                    print(f"   Re-rank Score: {result['rerank_score']:.2f}")
                else:
                    print(f"\n{i}. 📄 {result['source_file']} (chunk {result['chunk_index']})")
                    print(f"   Similarity Score: {score:.3f}")
                
                print(f"\n   📝 TEXT:")
                text = result['text']
                # Print text with line wrapping
                for line in text.split('\n'):
                    if line.strip():
                        print(f"      {line}")
                print()
            
            print("="*70)
            
        except KeyboardInterrupt:
            print("\n\n✓ Test interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
