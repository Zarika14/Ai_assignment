#!/usr/bin/env python3
"""
Index Documents Script

Run this script to index all policy documents in the rag/documents/ directory.
Generates:
  - rag/faiss_index.bin (FAISS vector index)
  - rag/chunks_metadata.json (chunk metadata)
"""

import sys
from pathlib import Path
from pipeline import RAGPipeline, print_chunk_stats

def main():
    """Run the full indexing pipeline."""
    print("\n" + "="*60)
    print("DOCUMENT INDEXING PIPELINE")
    print("="*60)
    
    try:
        # Initialize pipeline
        pipeline = RAGPipeline(rerank=True)
        
        # Load documents
        chunks = pipeline.load_documents()
        print(f"\n✓ Loaded and chunked documents")
        
        # Print statistics
        print_chunk_stats(chunks)
        
        # Build FAISS index
        pipeline.build_index()
        print(f"✓ Built FAISS index")
        
        # Save index and metadata
        pipeline.save_index()
        print(f"✓ Saved index and metadata")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"INDEXING COMPLETE")
        print(f"{'='*60}")
        print(f"Indexed {len(chunks)} chunks from {len(set(c['source_file'] for c in chunks))} documents")
        print(f"✓ Index: rag/faiss_index.bin")
        print(f"✓ Metadata: rag/chunks_metadata.json")
        print(f"Ready for retrieval!\n")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
