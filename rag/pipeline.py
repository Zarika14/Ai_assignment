"""
RAG Pipeline for Insurance Policy Documents

This module implements a complete retrieval-augmented generation pipeline:
- Custom text chunking with metadata tracking
- BGE embeddings with query prefix
- FAISS vector store (IndexFlatIP for normalized embeddings)
- Retrieval with optional cross-encoder re-ranking
- Grounded answer generation with source attribution

Example:
    >>> pipeline = RAGPipeline(rerank=True)
    >>> pipeline.load_documents()
    >>> pipeline.build_index()
    >>> pipeline.save_index()
    >>> result = pipeline.answer_with_sources("What is covered?")
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import requests
import logging

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

from sentence_transformers import SentenceTransformer, CrossEncoder

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Configuration
CHUNK_SIZE = 400  # tokens (approximate)
CHUNK_OVERLAP = 80  # tokens
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Paths
DOCS_DIR = Path(__file__).parent / "documents"
INDEX_PATH = Path(__file__).parent / "faiss_index.bin"
METADATA_PATH = Path(__file__).parent / "chunks_metadata.json"


class TextChunker:
    """
    Recursive character-based text splitter.
    
    Rationale:
    - 400 tokens fits within typical context windows (e.g., 2048 token limit)
      while preserving semantic units (paragraphs/sections)
    - 80 token overlap (20%) prevents context loss at chunk boundaries
    - Recursive splitting maintains logical document structure
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count: ~1 token per 4 characters (rough approximation)"""
        return len(text) // 4
    
    def split(self, text: str, source_file: str) -> List[Dict]:
        """
        Split text into chunks with metadata.
        
        Returns:
            List of dicts with keys: text, source_file, chunk_index, char_start, char_end
        """
        chunks = []
        char_pos = 0
        chunk_index = 0
        
        # Split by sentences (periods) first, then combine into chunks
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        current_chunk = ""
        chunk_start_pos = 0
        
        for sentence in sentences:
            sentence = sentence + "."
            
            # Check if adding this sentence would exceed chunk size
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            tokens = self.estimate_tokens(test_chunk)
            
            if tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_end_pos = char_pos
                chunks.append({
                    "text": current_chunk.strip(),
                    "source_file": source_file,
                    "chunk_index": chunk_index,
                    "char_start": chunk_start_pos,
                    "char_end": chunk_end_pos,
                })
                chunk_index += 1
                
                # Start new chunk with overlap
                # Find overlap by taking last N tokens
                overlap_tokens = int(self.overlap / 4)  # approx chars for overlap
                current_chunk = current_chunk[-overlap_tokens:] + " " + sentence
                chunk_start_pos = char_pos - overlap_tokens
            else:
                current_chunk = test_chunk if not current_chunk else test_chunk
            
            char_pos += len(sentence) + 1
        
        # Save final chunk
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "source_file": source_file,
                "chunk_index": chunk_index,
                "char_start": chunk_start_pos,
                "char_end": char_pos,
            })
        
        return chunks


class RAGPipeline:
    """Complete RAG pipeline with embedding, retrieval, and answer generation."""
    
    def __init__(self, rerank: bool = True):
        """
        Initialize the RAG pipeline.
        
        Args:
            rerank: If True, use cross-encoder re-ranker on retrieved candidates
        """
        self.rerank = rerank
        self.chunks = []
        self.embeddings = None
        self.index = None
        
        # Load models
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        if self.rerank:
            logger.info("Loading re-ranker model: %s", RERANKER_MODEL)
            self.reranker = CrossEncoder(RERANKER_MODEL)
        else:
            self.reranker = None
    
    def load_documents(self, docs_dir: Path = DOCS_DIR) -> List[Dict]:
        """Load and chunk all documents from the documents directory.
        
        Args:
            docs_dir: Path to directory containing .txt files
            
        Returns:
            List of chunk dictionaries
            
        Raises:
            FileNotFoundError: If documents directory doesn't exist
            ValueError: If no documents found in directory
        """
        # Validate directory exists
        if not docs_dir.exists():
            raise FileNotFoundError(
                f"Documents directory not found: {docs_dir}\n"
                f"Create policy files in: {docs_dir.resolve()}"
            )
        
        # Find document files
        doc_files = sorted([f for f in docs_dir.glob("*.txt")])
        
        if not doc_files:
            raise ValueError(
                f"No .txt files found in {docs_dir}\n"
                f"Expected files: policy_*.txt"
            )
        
        chunker = TextChunker()
        all_chunks = []
        
        logger.info("Loading %d documents from %s", len(doc_files), DOCS_DIR)
        
        for doc_file in doc_files:
            try:
                logger.info("  Processing: %s", doc_file.name)
                text = doc_file.read_text(encoding='utf-8')
                
                if not text.strip():
                    logger.warning(f"Warning: {doc_file.name} is empty, skipping")
                    continue
                
                chunks = chunker.split(text, doc_file.name)
                all_chunks.extend(chunks)
                logger.info("  %s -> %d chunks", doc_file.name, len(chunks))
            except Exception as e:
                logger.error(f"Error processing {doc_file.name}: {e}")
                raise
        
        if not all_chunks:
            raise ValueError("No chunks created from documents. Check document content.")
        
        self.chunks = all_chunks
        logger.info("Total chunks loaded: %d", len(self.chunks))
        return self.chunks
    
    def build_index(self) -> None:
        """Build FAISS index from chunks.
        
        Raises:
            ValueError: If no chunks loaded
        """
        if not self.chunks:
            raise ValueError(
                "No chunks loaded. Call load_documents() first.\n"
                "Example: pipeline.load_documents()"
            )
        
        logger.info("Embedding %d chunks...", len(self.chunks))
        texts = [chunk["text"] for chunk in self.chunks]
        
        try:
            # Embed all chunks (without BGE prefix for documents)
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            self.embeddings = embeddings
            
            # Create FAISS index (IndexFlatIP for cosine similarity with normalized embeddings)
            # BGE embeddings are already normalized
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings.astype(np.float32))
            
            logger.info("Index built: %d vectors (dim=%d)", self.index.ntotal, dimension)
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise ValueError(f"Index building failed: {e}")
    
    def save_index(self, index_path: Path = INDEX_PATH, metadata_path: Path = METADATA_PATH) -> None:
        """Save FAISS index and chunk metadata.
        
        Args:
            index_path: Where to save the FAISS index
            metadata_path: Where to save chunk metadata
            
        Raises:
            ValueError: If index not built
            IOError: If save fails
        """
        if self.index is None:
            raise ValueError(
                "Index not built. Call build_index() first.\n"
                "Example: pipeline.build_index()"
            )
        
        try:
            # Ensure parent directories exist
            index_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            logger.info("Index saved to: %s", index_path)
            
            # Save metadata
            metadata = [
                {
                    "source_file": chunk["source_file"],
                    "chunk_index": chunk["chunk_index"],
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                }
                for chunk in self.chunks
            ]
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info("Metadata saved to: %s", metadata_path)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise IOError(f"Failed to save index: {e}")
    
    def load_index(self, index_path: Path = INDEX_PATH, metadata_path: Path = METADATA_PATH) -> None:
        """Load FAISS index and chunk metadata.
        
        Args:
            index_path: Path to saved FAISS index
            metadata_path: Path to chunk metadata
            
        Raises:
            FileNotFoundError: If index or metadata files not found
        """
        # Validate files exist
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}\n"
                f"Run: python index_documents.py"
            )
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                f"Run: python index_documents.py"
            )
        
        # Validate documents directory exists
        if not DOCS_DIR.exists():
            raise FileNotFoundError(
                f"Documents directory not found: {DOCS_DIR}"
            )
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            logger.info("Index loaded from: %s", index_path)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Reconstruct chunks with full data from documents
            self.chunks = []
            chunker = TextChunker()
            
            for doc_file in sorted(DOCS_DIR.glob("*.txt")):
                text = doc_file.read_text(encoding='utf-8')
                chunks = chunker.split(text, doc_file.name)
                for chunk in chunks:
                    self.chunks.append(chunk)
            
            logger.info("Metadata loaded: %d chunks", len(metadata))
            
            if len(self.chunks) != self.index.ntotal:
                logger.warning(
                    f"Chunk count mismatch: {len(self.chunks)} chunks but "
                    f"index has {self.index.ntotal} vectors. "
                    f"Consider rebuilding the index."
                )
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = 3, rerank: Optional[bool] = None) -> List[Dict]:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: The search query (cannot be empty)
            top_k: Number of top results to return (must be > 0)
            rerank: Override self.rerank for this call (True/False/None)
        
        Returns:
            List of dicts with keys: text, source_file, chunk_index, similarity_score
            
        Raises:
            ValueError: If index not loaded, query invalid, or top_k invalid
            RuntimeError: If reranking fails
        """
        # Validate inputs
        if self.index is None:
            raise ValueError(
                "Index not loaded. Call load_index() first.\n"
                "Example: pipeline.load_index()"
            )
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be positive integer, got: {top_k}")
        
        if top_k > self.index.ntotal:
            logger.warning(
                f"top_k ({top_k}) exceeds total chunks ({self.index.ntotal}), "
                f"using {self.index.ntotal}"
            )
            top_k = self.index.ntotal
        
        try:
            # Add BGE query prefix
            prefixed_query = BGE_QUERY_PREFIX + query
            query_embedding = self.embedding_model.encode(prefixed_query)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Retrieve top_k * 3 candidates for re-ranking
            retrieve_k = top_k * 3 if (rerank if rerank is not None else self.rerank) else top_k
            retrieve_k = min(retrieve_k, self.index.ntotal)  # Don't request more than available
            
            scores, indices = self.index.search(query_embedding, retrieve_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:  # Valid index
                    chunk = self.chunks[int(idx)]
                    results.append({
                        "text": chunk["text"],
                        "source_file": chunk["source_file"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity_score": float(score),  # Cosine similarity (0-1)
                    })
            
            # Re-ranking if enabled
            use_rerank = rerank if rerank is not None else self.rerank
            if use_rerank:
                if self.reranker is None:
                    raise ValueError("Re-ranker not initialized. Set rerank=True during init.")
                
                logger.debug("Re-ranking %d candidates", len(results))
                
                try:
                    # Prepare query-document pairs for cross-encoder
                    pairs = [(query, result["text"]) for result in results]
                    rerank_scores = self.reranker.predict(pairs)
                    
                    # Add re-rank scores and sort
                    for result, score in zip(results, rerank_scores):
                        result["rerank_score"] = float(score)
                    
                    results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
                    results = results[:top_k]
                except Exception as e:
                    logger.error(f"Reranking failed: {e}")
                    raise RuntimeError(f"Reranking failed: {e}")
            else:
                results = results[:top_k]
            
            return results
            
        except ValueError:
            raise  # Re-raise validation errors
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise
    
    def answer_with_sources(self, query: str, top_k: int = 3, model_server_url: str = "http://localhost:8000", stream: bool = False) -> Dict:
        """Generate a grounded answer using retrieved chunks and the model server.
        
        IMPORTANT: Every answer includes inline source citations in the format [filename, chunk N]
        placed immediately after each fact they support.
        
        Args:
            query: The user query (cannot be empty)
            top_k: Number of chunks to retrieve (must be > 0)
            model_server_url: URL of the model server (must be valid)
            stream: If True, stream tokens as they're generated
            
        Returns:
            Dict with keys:
                - answer: Grounded answer with inline citations [filename, chunk N]
                - sources: List of source documents with scores
                - query: The original query
                - error: Error message (if applicable)
            
        Example:
            >>> result = pipeline.answer_with_sources("What is the deductible?")
            >>> print(result['answer'])
            "The collision deductible is $500 [policy_auto_comprehensive.txt, chunk 3]."
            >>> print(result['sources'])
            [{'source_file': 'policy_auto_comprehensive.txt', 'chunk_index': 3, ...}]
            
        Raises:
            ValueError: If inputs invalid or query empty
            ConnectionError: If model server unreachable
            TimeoutError: If model server doesn't respond in time
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be positive integer, got: {top_k}")
        
        # Validate model_server_url
        try:
            from urllib.parse import urlparse
            parsed = urlparse(model_server_url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError
        except:
            raise ValueError(f"Invalid model_server_url: {model_server_url}")
        
        # Retrieve relevant chunks
        logger.info("Retrieving top-%d chunks for: %s", top_k, query[:60])
        try:
            chunks = self.retrieve(query, top_k=top_k)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise
        
        if not chunks:
            logger.warning("No relevant chunks found for query")
            return {
                "query": query,
                "answer": "(No relevant information found in policies)",
                "sources": [],
            }
        
        # Build context from chunks
        context = ""
        sources = []
        for chunk in chunks:
            context += f"\n[{chunk['source_file']}, chunk {chunk['chunk_index']}]: {chunk['text']}\n"
            sources.append({
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
                "similarity_score": chunk.get("similarity_score", 0.0),
            })
        
        # Build prompt for model
        system_prompt = (
            "You are an insurance assistant. Answer based ONLY on the provided policy excerpts below.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. EVERY sentence or claim MUST include a source citation in the format [filename, chunk N]\n"
            "2. Place citations immediately after the fact they support\n"
            "3. Example: 'The deductible is $500 [policy_auto_comprehensive.txt, chunk 3].'\n"
            "4. If multiple sources support a claim, cite all of them: [file1.txt, chunk 1] and [file2.txt, chunk 2]\n"
            "5. DO NOT make claims without citations\n"
            "6. If information is not in the policies, explicitly say: 'This information is not covered in the provided policies.'\n"
            "7. Do not invent or assume information not in the provided excerpts"
        )
        
        user_message = f"""Based on ONLY the policy excerpts provided below, answer this question:

Question: {query}

POLICY EXCERPTS:
{context}

IMPORTANT REQUIREMENTS:
- Answer using ONLY information from the policy excerpts above
- EVERY claim must cite its source: [filename, chunk N]
- Place citations directly after the information they support
- Do not make any claims without citing the source
- If the answer is not found in the policies, explicitly state: 'This information is not covered in the provided policies.'

You must include the source attribution in brackets after EACH fact. For example:
- "The collision deductible is $500 [policy_auto_comprehensive.txt, chunk 3]."
- "Comprehensive coverage includes theft [policy_auto_comprehensive.txt, chunk 4] and vandalism [policy_auto_comprehensive.txt, chunk 4]."

Answer (with inline citations for every claim):"""
        
        # Validate model server connectivity
        logger.info("Connecting to model server: %s", model_server_url)
        try:
            health_response = requests.get(
                f"{model_server_url}/health",
                timeout=5
            )
            if health_response.status_code != 200:
                raise ConnectionError(
                    f"Model server returned status {health_response.status_code}"
                )
            logger.info("Model server is running")
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Model server at {model_server_url} is not responding (timeout)"
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to model server at {model_server_url}\n"
                f"Make sure: 1. Ollama is running (ollama serve)\n"
                f"           2. Model server is running (python main.py)"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to verify model server: {e}")
        
        # Call model server
        endpoint = "chat/stream" if stream else "chat"
        logger.info("Generating grounded answer for: %s", query[:60])
        
        try:
            if stream:
                # Use streaming endpoint
                answer = ""
                try:
                    with requests.post(
                        f"{model_server_url}/chat/stream",
                        json={
                            "message": user_message,
                            "system_prompt": system_prompt
                        },
                        stream=True,
                        timeout=600
                    ) as response:
                        if response.status_code != 200:
                            raise requests.HTTPError(
                                f"Model server returned status {response.status_code}"
                            )
                        
                        line_count = 0
                        for line in response.iter_lines():
                            if not line:
                                continue
                            
                            line_count += 1
                            
                            # Convert bytes to str if necessary
                            if isinstance(line, bytes):
                                try:
                                    line = line.decode('utf-8')
                                except UnicodeDecodeError:
                                    logger.warning(f"Failed to decode line {line_count}")
                                    continue
                            
                            # Parse SSE format: "data: {...}"
                            if not line.startswith("data:"):
                                continue
                            
                            try:
                                json_str = line[5:].strip()
                                if not json_str:
                                    continue
                                
                                data = json.loads(json_str)
                                
                                # Check for errors
                                if "error" in data and data["error"]:
                                    logger.error(f"Server error: {data['error']}")
                                    raise requests.HTTPError(f"Server error: {data['error']}")
                                
                                token = data.get("token", "")
                                done = data.get("done", False)
                                
                                if token:
                                    answer += token
                                logger.debug("Stream token: %s", token[:20] if token else "[empty]")
                                
                                if done:
                                    break
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse SSE event at line {line_count}: {repr(line[:100])}")
                                continue
                        
                        if line_count == 0:
                            raise RuntimeError("No data received from streaming endpoint")
                        
                    logger.debug("Stream complete")
                        
                except requests.exceptions.Timeout:
                    raise TimeoutError("Model server streaming response timed out")
                except requests.exceptions.ConnectionError as e:
                    raise ConnectionError(f"Connection lost during streaming: {e}")
            else:
                # Use non-streaming endpoint
                try:
                    response = requests.post(
                        f"{model_server_url}/chat",
                        json={
                            "message": user_message,
                            "system_prompt": system_prompt
                        },
                        timeout=600
                    )
                    response.raise_for_status()
                    result = response.json()
                    answer = result.get("response", "")
                    
                    if not answer:
                        logger.warning("Model server returned empty response")
                        answer = "(No response from model)"
                except requests.exceptions.Timeout:
                    raise TimeoutError("Model server response timed out")
                except requests.exceptions.ConnectionError as e:
                    raise ConnectionError(f"Connection error: {e}")
                except requests.exceptions.RequestException as e:
                    raise requests.RequestException(f"Request failed: {e}")
        
        except (TimeoutError, ConnectionError, requests.RequestException) as e:
            logger.error(f"Model server error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during answer generation: {e}")
            raise
        
        if not answer or not answer.strip():
            logger.warning("Generated answer is empty")
            answer = "(Failed to generate answer)"
        
        return {
            "query": query,
            "answer": answer.strip(),
            "sources": sources,
        }


def print_chunk_stats(chunks: List[Dict]) -> None:
    """Print statistics about chunk distribution."""
    from collections import Counter
    
    sources = Counter(c["source_file"] for c in chunks)
    sizes = [len(c["text"]) for c in chunks]
    logger.info(
        "Chunk statistics: total=%d, min_chars=%d, max_chars=%d, avg_chars=%.0f, by_source=%s",
        len(chunks),
        min(sizes) if sizes else 0,
        max(sizes) if sizes else 0,
        (sum(sizes) / len(sizes)) if sizes else 0,
        dict(sorted(sources.items())),
    )


if __name__ == "__main__":
    # Example usage
    pipeline = RAGPipeline(rerank=True)
    pipeline.load_documents()
    pipeline.build_index()
    pipeline.save_index()
    print_chunk_stats(pipeline.chunks)
