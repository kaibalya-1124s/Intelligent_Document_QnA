# app/core/text_splitter.py
from typing import List

def split_text_to_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of ~chunk_size words with `overlap` words overlap.
    Returns a list of strings (chunks).
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # move start forward but keep overlap
        start = end - overlap
        if start < 0:
            start = 0
        if start >= n:
            break
    return chunks
