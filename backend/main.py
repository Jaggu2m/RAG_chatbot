import os
import sys
import shutil
import time
from datetime import datetime
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path so rag_pipeline can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag_pipeline import ask_question

# ─── Load environment variables ─────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ─── MongoDB Setup ──────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI")
if MONGO_URI:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["rag_chatbot_db"]
    chats_collection = db["chats"]
else:
    print("WARNING: MONGO_URI not found in .env file. Chat history will not be saved.")
    chats_collection = None

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
    session_id: str
    question: str

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
    Main chat endpoint. 
    It fetches previous history from MongoDB using session_id,
    gets the RAG answer, and then saves the new interaction.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )


    try:
        # Load chat history from DB
        chat_history = []
        if chats_collection is not None:
            session_doc = chats_collection.find_one({"session_id": request.session_id})
            if session_doc and "messages" in session_doc:
                chat_history = session_doc["messages"]

        # Run pipeline
        result = ask_question(request.question, chat_history)

        # Save the new messages to DB
        if chats_collection is not None:
            new_messages = [
                {"role": "user", "content": request.question},
                {"role": "assistant", "content": result["answer"]}
            ]
            
            chats_collection.update_one(
                {"session_id": request.session_id},
                {
                    "$push": {"messages": {"$each": new_messages}},
                    "$setOnInsert": {"created_at": datetime.utcnow()},
                    "$set": {"updated_at": datetime.utcnow()}
                },
                upsert=True
            )

        return AnswerResponse(
            answer=result["answer"],
            sources=[SourceModel(**s) for s in result["sources"]]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

@app.get("/sessions")
def get_sessions():
    """Returns a list of all chat session documents, sorted by latest updated."""
    if chats_collection is None:
        return []
    
    docs = chats_collection.find({}, {"session_id": 1, "updated_at": 1, "messages": {"$slice": 1}}).sort("updated_at", -1)
    
    sessions = []
    for doc in docs:
        first_msg = doc["messages"][0]["content"] if "messages" in doc and doc["messages"] else "New Chat"
        # Truncate title
        title = first_msg[:30] + "..." if len(first_msg) > 30 else first_msg
        
        sessions.append({
            "session_id": doc["session_id"],
            "title": title
        })
        
    return sessions

@app.get("/chat/{session_id}")
def get_chat_history(session_id: str):
    """Returns the full message history and sources for a specific session."""
    if chats_collection is None:
        return {"messages": []}
        
    doc = chats_collection.find_one({"session_id": session_id}, {"_id": 0})
    if not doc:
        return {"messages": []}
        
    return {"messages": doc.get("messages", [])}


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