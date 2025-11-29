from fastapi import APIRouter, UploadFile, File
from app.core.document_loader import extract_text_from_file
from app.core.text_splitter import split_text_to_chunks
from app.core.embeddings import embed_texts
from app.core.vector_store import FaissVectorStore
import numpy as np
import os

router = APIRouter()

# Store FAISS index in memory (simple approach)
VECTOR_STORE = None

@router.post("/")
async def ingest_document(file: UploadFile = File(...)):
    global VECTOR_STORE

    # Read the uploaded file
    content = await file.read()
    text = extract_text_from_file(file.filename, content)

    if not text.strip():
        return {"error": "Unable to extract text from file"}

    # Split into chunks
    chunks = split_text_to_chunks(text)
    embeddings = embed_texts(chunks)

    # Initialize vector store (384 dim for MiniLM)
    VECTOR_STORE = FaissVectorStore(dim=embeddings.shape[1])

    # Add all chunks with metadata
    metas = [
        {"source": file.filename, "chunk_id": i, "text": chunk}
        for i, chunk in enumerate(chunks)
    ]

    VECTOR_STORE.add(embeddings, metas)

    return {
        "message": "Document ingested successfully!",
        "chunks": len(chunks),
        "filename": file.filename
    }
