"""
Document embeddings and RAG (Retrieval-Augmented Generation) search.
Uses sentence-transformers for embeddings and FAISS for vector search.
Falls back to simple TF-IDF if heavy dependencies unavailable.
"""

from typing import List, Dict, Any
import math
import re
from collections import Counter


class DocumentEmbedder:
    """
    Manages document embeddings for semantic search (RAG).
    
    Tries to use sentence-transformers + FAISS for true semantic search.
    Falls back to BM25-style keyword search if not available.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.use_semantic = False
        self.chunks: List[Dict[str, Any]] = []   # {doc_name, chunk, chunk_id, embedding}
        self.index = None
        self.encoder = None

        # Try to initialize semantic search
        self._try_init_semantic()

    def _try_init_semantic(self):
        """Attempt to load sentence-transformers and FAISS."""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np

            self.encoder = SentenceTransformer(self.embedding_model_name)
            self.use_semantic = True
        except ImportError:
            # Graceful fallback to keyword search
            self.use_semantic = False
        except Exception:
            self.use_semantic = False

    def add_document(self, doc_name: str, text: str, chunk_size: int = 500):
        """
        Add a document by splitting it into chunks and computing embeddings.
        """
        if not text or not text.strip():
            return

        # Split into chunks
        raw_chunks = self._split_into_chunks(text, chunk_size)

        if self.use_semantic:
            self._add_semantic_chunks(doc_name, raw_chunks)
        else:
            self._add_keyword_chunks(doc_name, raw_chunks)

    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into overlapping chunks."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > chunk_size and current:
                chunks.append(" ".join(current))
                # Keep last sentence for overlap
                current = [current[-1]] if current else []
                current_len = len(current[0]) if current else 0
            current.append(sent)
            current_len += sent_len

        if current:
            chunks.append(" ".join(current))

        return [c for c in chunks if len(c.strip()) > 30]

    def _add_semantic_chunks(self, doc_name: str, raw_chunks: List[str]):
        """Add chunks with semantic embeddings."""
        try:
            import faiss
            import numpy as np

            embeddings = self.encoder.encode(raw_chunks, show_progress_bar=False)
            dim = embeddings.shape[1]

            if self.index is None:
                self.index = faiss.IndexFlatL2(dim)

            # Convert to float32
            emb_f32 = embeddings.astype(np.float32)
            self.index.add(emb_f32)

            start_id = len(self.chunks)
            for i, (chunk, emb) in enumerate(zip(raw_chunks, embeddings)):
                self.chunks.append({
                    "doc_name": doc_name,
                    "chunk": chunk,
                    "chunk_id": start_id + i,
                    "embedding": emb,
                })
        except Exception:
            # Fallback if semantic fails mid-use
            self._add_keyword_chunks(doc_name, raw_chunks)

    def _add_keyword_chunks(self, doc_name: str, raw_chunks: List[str]):
        """Add chunks for keyword-based search (no embeddings)."""
        start_id = len(self.chunks)
        for i, chunk in enumerate(raw_chunks):
            self.chunks.append({
                "doc_name": doc_name,
                "chunk": chunk,
                "chunk_id": start_id + i,
                "embedding": None,
            })

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for the most relevant chunks given a query.
        Uses semantic search if available, else keyword (BM25-style) search.
        """
        if not self.chunks:
            return []

        if self.use_semantic and self.index is not None:
            return self._semantic_search(query, top_k)
        else:
            return self._keyword_search(query, top_k)

    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Vector similarity search using FAISS."""
        try:
            import numpy as np

            query_emb = self.encoder.encode([query]).astype(np.float32)
            k = min(top_k, len(self.chunks))
            distances, indices = self.index.search(query_emb, k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.chunks):
                    chunk_data = self.chunks[idx].copy()
                    chunk_data["score"] = float(1 / (1 + dist))   # Convert distance to score
                    chunk_data.pop("embedding", None)
                    results.append(chunk_data)
            return results
        except Exception:
            return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Simple TF-IDF-style keyword search fallback."""
        query_terms = set(re.sub(r'[^\w\s]', '', query.lower()).split())

        scored = []
        doc_counts = Counter(c["doc_name"] for c in self.chunks)
        n_docs = len(set(c["doc_name"] for c in self.chunks))

        for chunk_data in self.chunks:
            chunk_text = chunk_data["chunk"].lower()
            chunk_terms = re.sub(r'[^\w\s]', '', chunk_text).split()
            chunk_len = max(len(chunk_terms), 1)

            # BM25-like scoring
            score = 0.0
            for term in query_terms:
                tf = chunk_terms.count(term) / chunk_len
                # IDF approximation
                df = sum(1 for c in self.chunks if term in c["chunk"].lower())
                idf = math.log((n_docs + 1) / (df + 1)) + 1
                score += tf * idf

            if score > 0:
                result = {
                    "doc_name": chunk_data["doc_name"],
                    "chunk": chunk_data["chunk"],
                    "chunk_id": chunk_data["chunk_id"],
                    "score": score,
                }
                scored.append(result)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def clear(self):
        """Remove all stored chunks and reset the index."""
        self.chunks = []
        self.index = None
        if self.use_semantic:
            try:
                import faiss
                # Will be re-initialized on next add
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the embedder state."""
        doc_names = list(set(c["doc_name"] for c in self.chunks))
        return {
            "total_chunks": len(self.chunks),
            "documents": doc_names,
            "search_type": "semantic" if self.use_semantic else "keyword",
        }
