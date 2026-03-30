# RAG Pipeline - Part 2

## Overview

This directory contains a complete Retrieval-Augmented Generation (RAG) pipeline for insurance policy documents. The pipeline chunks documents, embeds them with sentence transformers, indexes them with FAISS, retrieves relevant chunks for queries, and generates grounded answers with source attribution.

## Architecture

### Step 1: Text Chunking

**Strategy**: Recursive character-based text splitting with metadata tracking.

**Parameters**:
- **Chunk Size**: 400 tokens (~1600 characters, approximate)
- **Overlap**: 80 tokens (20% of chunk size)

**Rationale**:
- 400 tokens fits comfortably within typical LLM context windows (e.g., 2048 tokens) while preserving semantic units
- Structured documents (policies) have clear paragraph/section boundaries; recursive splitting preserves these
- 80 token overlap (20%) prevents critical context loss at chunk boundaries
- Each chunk stores metadata: `source_file`, `chunk_index`, `char_start`, `char_end`

**Example chunk structure**:
```json
{
  "text": "Coverage Limit: Actual Cash Value\nDeductible: $500\n...",
  "source_file": "policy_auto_comprehensive.txt",
  "chunk_index": 3,
  "char_start": 1248,
  "char_end": 1648
}
```

### Step 2: Embedding with BGE

**Model**: `BAAI/bge-small-en-v1.5`

**Why BGE over other models?**
- **Specialized for retrieval**: Trained explicitly for semantic search and ranking tasks
- **Balanced efficiency**: Small variant (109M params) is fast while maintaining quality
- **State-of-the-art**: Outperforms other models on BEIR benchmark for dense retrieval
- **Matryoshka embedding**: Supports variable-length embeddings for efficiency

**Why `bge-small-en-v1.5` over `bge-large`?**
- **Inference speed**: Smaller model runs faster on CPU (relevant for this assignment)
- **Memory efficiency**: Fits easily on consumer hardware
- **Quality vs speed tradeoff**: Minimal performance drop for substantial speed gain
- **Production reality**: Small models sufficient for most retrieval tasks

**BGE Query Prefix**:
- Query: `"Represent this sentence for searching relevant passages: " + user_query`
- Document: Use raw text (no prefix)
- This asymmetric prefix is BGE-specific and improves retrieval quality

### Step 3: Vector Store with FAISS

**Index Type**: `IndexFlatIP` (Inner Product)

**Why Inner Product?**
- BGE embeddings are **normalized** (L2 norm = 1)
- Inner product of normalized vectors = cosine similarity
- Computationally efficient and numerically stable
- Similarity scores are naturally in range [0, 1]

**Index Operations**:
- Create: `faiss.IndexFlatIP(embedding_dim)`
- Add vectors: `index.add(embeddings)`
- Search: `index.search(query_embedding, top_k)`
- Persist: `faiss.write_index(index, path)` / `faiss.read_index(path)`

**Metadata Persistence**:
- FAISS stores only vectors; metadata stored separately in `chunks_metadata.json`
- Metadata includes: source file, chunk index, character positions
- Synchronized by array index

### Step 4: Retrieval Function

**Function**: `retrieve(query: str, top_k: int = 3, rerank: bool = True) -> List[Dict]`

**Process**:
1. Add BGE query prefix to user query
2. Embed query using embedding model
3. Retrieve top `k*3` candidates from FAISS (for re-ranking)
4. Extract similarity scores (inner product / cosine similarity)
5. Optionally re-rank with cross-encoder
6. Return top `k` results

**Output Structure**:
```python
[
  {
    "text": "...",
    "source_file": "policy_auto_comprehensive.txt",
    "chunk_index": 3,
    "similarity_score": 0.643,  # Cosine similarity [0, 1]
    "rerank_score": 5.2  # Optional: cross-encoder score
  },
  ...
]
```

### Step 5: Cross-Encoder Re-ranker (Bonus)

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**How it differs from bi-encoder (embedding model)**:
- **Bi-encoder** (BGE): Encodes query and documents separately, fast but less accurate
- **Cross-encoder**: Jointly encodes query-document pairs, slower but more accurate

**Re-ranking Process**:
1. Retrieve top `k*3` candidates using bi-encoder (fast initial filtering)
2. Compute query-document relevance scores using cross-encoder for each candidate
3. Re-rank by cross-encoder scores
4. Return top `k` re-ranked results

**Why this pipeline?**
- **Speed**: Bi-encoder fast filtering reduces cross-encoder computation (3x documents)
- **Accuracy**: Cross-encoder provides refined relevance scores
- Common pattern in production search systems (dense retrieval → re-ranking)

**Usage**:
```python
pipeline = RAGPipeline(rerank=True)  # Enable re-ranking
results = pipeline.retrieve(query, top_k=3)
```

### Step 6: Grounded Answer Generation with Inline Citations

**Function**: `answer_with_sources(query: str, top_k: int = 3) -> Dict`

**Key Feature**: Every answer includes inline source citations

**Process**:
1. Retrieve top `k` relevant chunks
2. Build context string with source attribution: `[filename, chunk N]: text`
3. Construct prompt with system + user message
4. Call model server (`/chat` endpoint) with retrieved context
5. Model generates answer with **inline citations** after every fact
6. Return answer with source metadata

**Inline Citations Format**:
- Place citation immediately after each fact
- Format: `[filename, chunk N]`
- Example: `"The collision deductible is $500 [policy_auto_comprehensive.txt, chunk 3]."`

**Prompt Structure** (Updated to enforce citations):
```
System: "You are an insurance assistant. Answer based ONLY on policy excerpts.
CRITICAL: EVERY sentence or claim MUST include a source citation in the format 
[filename, chunk N]. Place citations immediately after the fact they support.
Example: 'The deductible is $500 [policy_auto_comprehensive.txt, chunk 3].'
DO NOT make claims without citations."

User: "Based on ONLY the policy excerpts provided below, answer this question:
Question: What is the deductible for collision coverage?

POLICY EXCERPTS:
[policy_auto_comprehensive.txt, chunk 3]: Coverage Limit: Actual Cash Value...
Deductible: $500...

IMPORTANT: EVERY claim must cite its source: [filename, chunk N]
...
Answer (with inline citations for every claim):"
```

**Example Output**:
```
User Query: "What is covered in comprehensive auto insurance?"

LLM Answer with Inline Citations:
"Comprehensive auto insurance covers damage to your vehicle caused by events 
other than collision [policy_auto_comprehensive.txt, chunk 4]. This includes 
theft [policy_auto_comprehensive.txt, chunk 4] and vandalism [policy_auto_comprehensive.txt, chunk 4]. 
The policy also covers falling objects [policy_auto_comprehensive.txt, chunk 4] and 
weather events like hail, wind, and flood [policy_auto_comprehensive.txt, chunk 4]. 
The deductible for comprehensive coverage is $250 [policy_auto_comprehensive.txt, chunk 4]."

Sources:
1. policy_auto_comprehensive.txt, chunk 4 (score: 0.715)
```

**Return Format**:
```python
{
  "query": "What is the deductible for collision coverage?",
  "answer": "Based on [policy_auto_comprehensive.txt, chunk 3], ...",
  "sources": [
    {
      "source_file": "policy_auto_comprehensive.txt",
      "chunk_index": 3,
      "similarity_score": 0.715
    }
  ]
}
```

## Files

- `documents/` - Policy documents (3x ~600 word realistic insurance policies)
  - `policy_auto_comprehensive.txt` - Auto insurance with collision/comprehensive
  - `policy_home_standard.txt` - Homeowners with dwelling + personal property
  - `policy_health_bronze.txt` - Health insurance with deductibles/copays
- `pipeline.py` - Complete RAG pipeline implementation
- `index_documents.py` - Script to build and save index
- `faiss_index.bin` - FAISS vector index (generated)
- `chunks_metadata.json` - Chunk metadata (generated)

## Usage

### Index Documents

```bash
python rag/index_documents.py
```

Outputs:
- `rag/faiss_index.bin` - FAISS index with embeddings
- `rag/chunks_metadata.json` - Chunk metadata (source, index, positions)

**Example output**:
```
============================================================
DOCUMENT INDEXING PIPELINE
============================================================

Loading 3 documents...
  Processing: policy_auto_comprehensive.txt
    → 8 chunks
  Processing: policy_health_bronze.txt
    → 7 chunks
  Processing: policy_home_standard.txt
    → 9 chunks

Total chunks: 24

Embedding 24 chunks...
✓ Loaded and chunked documents

============================================================
CHUNK STATISTICS
============================================================
Total chunks: 24

Chunks by source:
  policy_auto_comprehensive.txt: 8 chunks
  policy_health_bronze.txt: 7 chunks
  policy_home_standard.txt: 9 chunks

Chunk size distribution:
  Min: 156 chars
  Max: 1842 chars
  Avg: 798 chars
============================================================
```

### Retrieve and Answer

```python
from rag.pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(rerank=True)
pipeline.load_index()

# Retrieve relevant chunks
results = pipeline.retrieve("What is the collision deductible?", top_k=3)
for result in results:
    print(f"[{result['source_file']}, chunk {result['chunk_index']}]")
    print(f"  Score: {result['similarity_score']:.3f}")
    print(f"  Text: {result['text'][:100]}...")

# Generate grounded answer WITH INLINE CITATIONS
answer = pipeline.answer_with_sources("What is the collision deductible?")

# Answer includes citations for EVERY fact
print("Answer:", answer['answer'])
print("Sources:", answer['sources'])
```

## Example Query + Retrieval + Answer with Inline Citations

**Query**: "What is covered in comprehensive auto insurance?"

**Retrieved Chunks**:
```
[policy_auto_comprehensive.txt, chunk 4]: Score 0.751
  "Comprehensive Coverage... This includes: theft, vandalism, falling objects, 
   weather events (hail, wind, flood), animal strikes, and fire..."

[policy_home_standard.txt, chunk 2]: Score 0.423
  "Dwelling Coverage... Protects the structure of the home..."

[policy_health_bronze.txt, chunk 1]: Score 0.312
  "Coverage Types... Preventive care, emergency services..."
```

**LLM Answer with Inline Citations** (EVERY fact is cited):
```
Comprehensive auto insurance covers damage to your vehicle caused by events 
other than collision [policy_auto_comprehensive.txt, chunk 4]. Specifically, 
it includes theft [policy_auto_comprehensive.txt, chunk 4], vandalism [policy_auto_comprehensive.txt, chunk 4], 
falling objects [policy_auto_comprehensive.txt, chunk 4], weather events such as hail, 
wind, and flood [policy_auto_comprehensive.txt, chunk 4], animal strikes [policy_auto_comprehensive.txt, chunk 4], 
and fire [policy_auto_comprehensive.txt, chunk 4]. The deductible for comprehensive coverage 
is $250 per claim [policy_auto_comprehensive.txt, chunk 4].
```

**Key Points**:
- ✅ **Every sentence cited** - Each fact shows its source immediately after
- ✅ **Consistent format** - [filename, chunk N] for easy reference
- ✅ **Multiple cites allowed** - If multiple sources support a claim, all are cited
- ✅ **Transparent traceability** - Users know exactly where each fact comes from

## Performance Notes

- **Indexing**: ~2-5 seconds for 3 policies (~2000 words) on CPU
- **Retrieval**: ~500ms (bi-encoder) + ~200ms (cross-encoder re-ranking)
- **Answer generation**: ~2-5 seconds (depends on model server latency)
- **Memory**: ~100MB for FAISS index + model weights

## Error Handling & Validation

### Input Validation

All methods validate inputs and provide clear error messages:

```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()

# ❌ ERROR: Empty query
try:
    pipeline.retrieve("")
except ValueError as e:
    print(f"Error: {e}")  # "Query cannot be empty"

# ❌ ERROR: Invalid top_k
try:
    pipeline.retrieve("test query", top_k=-1)
except ValueError as e:
    print(f"Error: {e}")  # "top_k must be positive integer"

# ❌ ERROR: No documents loaded
try:
    pipeline.build_index()
except ValueError as e:
    print(f"Error: {e}")  # "No chunks loaded. Call load_documents() first."
```

### File Validation

The pipeline validates file existence before operations:

```python
# ❌ ERROR: Documents directory not found
try:
    pipeline.load_documents()
except FileNotFoundError as e:
    print(f"Error: {e}")

# ❌ ERROR: Index files not found
try:
    pipeline.load_index()
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Run: python index_documents.py")
```

### Model Server Connectivity

The pipeline validates model server availability before calling:

```python
# ✓ Checks if server is running
# ✓ Validates response status
# ✓ Handles timeout gracefully

try:
    answer = pipeline.answer_with_sources(
        "What is covered?",
        model_server_url="http://localhost:8000"
    )
except ConnectionError as e:
    print(f"Model server error: {e}")
    print("Make sure: 1. ollama serve")
    print("           2. python main.py")
except TimeoutError as e:
    print(f"Model server timeout: {e}")
```

### Streaming Error Handling

Streaming mode handles connection issues and malformed responses:

```python
try:
    answer = pipeline.answer_with_sources(
        query="What is covered?",
        stream=True  # Token-by-token streaming
    )
except (TimeoutError, ConnectionError) as e:
    print(f"Streaming failed: {e}")
    # Gracefully falls back or raises with meaningful message
```

### Chunk Mismatch Detection

Warns if index vectors don't match loaded chunks:

```python
# If index and documents get out of sync:
# ⚠️ "Chunk count mismatch: 25 chunks but index has 24 vectors"
# Suggestion: "Consider rebuilding the index"
```

## Testing

### Unit Tests

```bash
# Run comprehensive test suite
python rag/test_comprehensive.py
```

Tests cover:
- TextChunker functionality (empty text, single sentence, long text)
- RAGPipeline initialization
- Error handling (all ValueError cases)
- Input validation (empty queries, invalid top_k)
- Model integration

### End-to-End Tests

```bash
# Interactive retrieval testing
python rag/test_interactive.py

# Grounded answer generation
python rag/test_grounded_answer.py
```

## Troubleshooting

### "No .txt files found in documents/"
- Make sure policy documents exist in `rag/documents/`
- Files must be named `*.txt`

### "Index not found: faiss_index.bin"
- Run: `python rag/index_documents.py`
- This creates the index from documents

### "Cannot connect to model_server"
- Start Ollama: `ollama serve`
- Start model server: `python rag/../model_server/main.py`
- Verify: `curl http://localhost:8000/health`

### "Chunk count mismatch" warning
- Documents or policy files were modified after indexing
- Solution: `python rag/index_documents.py` (rebuild index)

### Empty retrieval results
- Query might be too specific or documents don't contain relevant information
- Try broader queries: "What coverage types are available?"
- Check if policy documents contain policy text (not just headers)

## Architecture Decisions

### Why Sentence-Transformers + FAISS?
- Lightweight, no external services
- Works offline on CPU
- Production-proven architecture

### Why BGE Embeddings?
- Optimized specifically for retrieval tasks
- Normalized embeddings = direct cosine similarity
- Strong performance on BEIR benchmark

### Why Cross-Encoder Re-ranking?
- Significant relevance improvement over bi-encoder alone
- Re-rank only top-k*3 candidates (efficient, not all vectors)
- Common pattern: dense retrieval (fast) → re-ranking (accurate)

### Why Recursive Text Chunking?
- Preserves document structure (paragraphs, sections)
- 400 tokens = good for LLM context windows
- 20% overlap prevents context loss at boundaries

## API Reference

### RAGPipeline

```python
class RAGPipeline:
    def __init__(self, rerank: bool = True)
        """Initialize pipeline with embedding and reranking models."""
    
    def load_documents(self, docs_dir: Path) -> List[Dict]
        """Load and chunk all .txt files from directory."""
    
    def build_index(self) -> None
        """Build FAISS index from chunks."""
    
    def save_index(self, index_path: Path, metadata_path: Path) -> None
        """Save index and metadata to disk."""
    
    def load_index(self, index_path: Path, metadata_path: Path) -> None
        """Load index and metadata from disk."""
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 3, 
        rerank: Optional[bool] = None
    ) -> List[Dict]
        """Retrieve top-k relevant chunks (with optional re-ranking)."""
    
    def answer_with_sources(
        self,
        query: str,
        top_k: int = 3,
        model_server_url: str = "http://localhost:8000",
        stream: bool = False
    ) -> Dict
        """Generate grounded answer with source attribution."""
```

### TextChunker

```python
class TextChunker:
    def __init__(self, chunk_size: int = 400, overlap: int = 80)
        """Initialize chunker with token size and overlap."""
    
    def split(self, text: str, source_file: str) -> List[Dict]
        """Split text into chunks with metadata."""
    
    def estimate_tokens(self, text: str) -> int
        """Estimate token count (1 token ≈ 4 characters)."""
```

## Dependencies

- `sentence-transformers` - BGE embeddings and cross-encoder
- `faiss-cpu` - Vector search
- `numpy` - Numeric operations
- `requests` - HTTP calls to model server
