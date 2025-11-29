from fastapi import FastAPI
from app.routers import ingest, qa

# Create FastAPI app
app = FastAPI(
    title="Intelligent Document QnA System",
    description="Upload documents and ask questions using AI",
    version="1.0.0"
)

# Include Router Endpoints
app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
app.include_router(qa.router, prefix="/qa", tags=["Question-Answering"])

# Root API Check
@app.get("/")
def home():
    return {"message": "Intelligent Document QnA System is running!"}
