# app/core/embeddings.py
from typing import List
import numpy as np

_model = None

def get_embedding_model(name: str = "all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(name)
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Return numpy array of embeddings for a list of texts.
    """
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    model = get_embedding_model()
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embs.astype("float32")
