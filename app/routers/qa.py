from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def ask_question(q: str):
    return {"answer": f"You asked: {q}"}
