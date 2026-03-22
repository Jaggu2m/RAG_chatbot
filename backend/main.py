import os
import sys
import shutil
import time
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path so rag_pipeline can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag_pipeline import ask_question

# ─── Create FastAPI app ──────────────────────────────────────────────
app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval Augmented Generation chatbot using Groq + Pinecone",
    version="1.0.0"
)

# ─── Allow Streamlit frontend to talk to this backend ───────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request and response models ────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    chat_history: list[dict] = []

class SourceModel(BaseModel):
    file: str
    content: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceModel]

# ─── Routes ─────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "RAG Chatbot API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest):
    """
    Main chat endpoint. Receives a question and optional chat history,
    runs it through the RAG pipeline, and returns the answer with sources.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        result = ask_question(request.question, request.chat_history)
        return AnswerResponse(
            answer=result["answer"],
            sources=[SourceModel(**s) for s in result["sources"]]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a document upload from the frontend,
    saves it to /docs, and re-indexes it into Pinecone.
    """
    # Only allow supported file types
    allowed_types = [".pdf", ".txt", ".md", ".csv"]
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_types}"
        )

    try:
        # Save uploaded file to /docs folder
        save_path = os.path.join("docs", file.filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Re-index the new file into Pinecone
        from ingest import ingest_single_file
        chunks = ingest_single_file(save_path)

        # Wait for Pinecone to sync the new vectors
        time.sleep(5)

        return {
            "message": f"'{file.filename}' uploaded and indexed successfully.",
            "chunks": chunks,
            "file": file.filename
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file: {str(e)}"
        )