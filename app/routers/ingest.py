from fastapi import APIRouter

router = APIRouter()

@router.post("/upload")
def upload_file():
    return {"message": "Ingestion endpoint working!"}
