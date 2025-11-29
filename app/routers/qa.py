from fastapi import APIRouter, HTTPException
from app.core.embeddings import embed_texts
from app.core.llm import generate_answer
from app.routers.ingest import VECTOR_STORE

router = APIRouter()

@router.get("/")
async def ask_question(question: str):
    if VECTOR_STORE is None:
        raise HTTPException(status_code=400, detail="No document ingested yet!")

    # Embed question
    q_emb = embed_texts([question])[0]

    # Search vector DB
    results = VECTOR_STORE.search(q_emb, top_k=3)

    if not results:
        return {"answer": "No relevant information found"}

    contexts = [r["text"] for r in results]

    # Generate final answer
    answer = generate_answer(question, contexts)

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts
    }
