# app/core/vector_store.py
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Optional

class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        # simple L2 index
        self.index = faiss.IndexFlatL2(dim)
        self.metadatas: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        """
        vectors: np.ndarray of shape (n, dim)
        metadatas: list of metadata dicts length n (e.g., {'source': filename, 'chunk_id': i, 'text': chunk})
        """
        if vectors.size == 0:
            return
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")
        self.index.add(vectors)
        self.metadatas.extend(metadatas)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype("float32")
        D, I = self.index.search(query_vector.reshape(1, -1), top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.metadatas):
                results.append(self.metadatas[idx])
        return results

    def save(self, path_index: str, path_meta: str):
        faiss.write_index(self.index, path_index)
        with open(path_meta, "wb") as f:
            pickle.dump(self.metadatas, f)

    @classmethod
    def load(cls, path_index: str, path_meta: str) -> "FaissVectorStore":
        # metadata pickled should contain dim info implicitly, but we must load index first
        index = faiss.read_index(path_index)
        dim = index.d
        store = cls(dim)
        store.index = index
        with open(path_meta, "rb") as f:
            store.metadatas = pickle.load(f)
        return store
