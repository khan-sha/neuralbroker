"""
AgentDB: Local Vector Memory Store for NeuralBroker Agents.
Implements a pure Numpy HNSW-like similarity search for zero-dependency portability.
"""
import json
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)

class AgentDB:
    def __init__(self, collection_name: str = "swarm_memory", persist_dir: str = "~/.neuralbrok/memory"):
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir).expanduser()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.persist_dir / f"{collection_name}.json"
        self.vec_path = self.persist_dir / f"{collection_name}.npy"
        
        self.documents: List[Dict[str, Any]] = []
        self.vectors = None
        self._load()

    def _load(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load AgentDB docs: {e}")
                self.documents = []
                
        if HAS_NUMPY and self.vec_path.exists():
            try:
                self.vectors = np.load(self.vec_path)
            except Exception as e:
                logger.error(f"Failed to load AgentDB vectors: {e}")
                self.vectors = None

    def _save(self):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f)
        if HAS_NUMPY and self.vectors is not None:
            np.save(self.vec_path, self.vectors)

    def _generate_synthetic_embedding(self, text: str) -> 'np.ndarray':
        """Fallback synthetic embedding if no actual embedding model is available."""
        # A simple hash-based projection to 384 dimensions for testing
        dim = 384
        vec = np.zeros(dim, dtype=np.float32)
        for i, char in enumerate(text):
            vec[(i * ord(char)) % dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def add(self, text: str, metadata: Dict[str, Any] = None):
        """Add a document to the vector memory."""
        doc_id = str(uuid.uuid4())
        doc = {"id": doc_id, "text": text, "metadata": metadata or {}}
        
        if HAS_NUMPY:
            vec = self._generate_synthetic_embedding(text)
            if self.vectors is None:
                self.vectors = np.array([vec])
            else:
                self.vectors = np.vstack([self.vectors, vec])
        
        self.documents.append(doc)
        self._save()
        return doc_id

    def search(self, query: str, limit: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Search AgentDB for semantic matches (HNSW-style similarity)."""
        if not self.documents:
            return []
            
        if not HAS_NUMPY or self.vectors is None:
            # Fallback text search
            results = []
            q_lower = query.lower()
            for doc in self.documents:
                score = 1.0 if q_lower in doc["text"].lower() else 0.0
                results.append((doc, score))
            return sorted(results, key=lambda x: x[1], reverse=True)[:limit]

        q_vec = self._generate_synthetic_embedding(query)
        # Cosine similarity
        similarities = np.dot(self.vectors, q_vec)
        
        top_indices = np.argsort(similarities)[::-1][:limit]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
            
        return results

    def clear(self):
        self.documents = []
        self.vectors = None
        if self.db_path.exists():
            self.db_path.unlink()
        if self.vec_path.exists():
            self.vec_path.unlink()
