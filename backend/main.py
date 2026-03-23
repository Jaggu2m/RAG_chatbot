import os
import sys
import shutil
import time
from datetime import datetime
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from authlib.integrations.starlette_client import OAuth
import jwt
from fastapi.responses import RedirectResponse

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

# ─── Auth Setup ──────────────────────────────────────────────────────
app.add_middleware(SessionMiddleware, secret_key=os.getenv("JWT_SECRET", "super-secret"))
oauth = OAuth()
oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

JWT_SECRET = os.getenv("JWT_SECRET", "super-secret")
ALGORITHM = "HS256"
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[ALGORITHM])
        return payload["sub"] # user's email
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ─── Auth Routes ─────────────────────────────────────────────────────
@app.get("/auth/login")
async def login(request: Request):
    redirect_uri = "http://localhost:8000/auth/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/callback")
async def auth_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        userinfo = token.get("userinfo")
        if not userinfo:
            raise HTTPException(status_code=400, detail="Could not access user info")
        
        user_email = userinfo.get("email")
        
        # Create a JWT token for our API
        jwt_token = jwt.encode(
            {"sub": user_email, "name": userinfo.get("name")},
            JWT_SECRET,
            algorithm=ALGORITHM
        )
        
        # Redirect back to Streamlit with the token
        return RedirectResponse(url=f"http://localhost:8501/?token={jwt_token}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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
def chat(request: QuestionRequest, user_email: str = Depends(verify_token)):
    """
    Main chat endpoint. 
    It fetches previous history from MongoDB using session_id + user_email,
    gets the RAG answer strictly from user documents, and saves the new interaction.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Load chat history from DB
        chat_history = []
        if chats_collection is not None:
            session_doc = chats_collection.find_one({
                "session_id": request.session_id,
                "user_email": user_email
            })
            if session_doc and "messages" in session_doc:
                chat_history = session_doc["messages"]

        # Run pipeline (pass user_email to filter pinecone vectors)
        result = ask_question(request.question, user_email, chat_history)

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
                    "$setOnInsert": {
                        "created_at": datetime.utcnow(),
                        "user_email": user_email
                    },
                    "$set": {"updated_at": datetime.utcnow()}
                },
                upsert=True
            )

        return AnswerResponse(
            answer=result["answer"],
            sources=[SourceModel(**s) for s in result["sources"]]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/sessions")
def get_sessions(user_email: str = Depends(verify_token)):
    """Returns a list of all chat session documents for the user."""
    if chats_collection is None:
        return []
    
    docs = chats_collection.find(
        {"user_email": user_email}, 
        {"session_id": 1, "updated_at": 1, "messages": {"$slice": 1}}
    ).sort("updated_at", -1)
    
    sessions = []
    for doc in docs:
        first_msg = doc["messages"][0]["content"] if "messages" in doc and doc["messages"] else "New Chat"
        title = first_msg[:30] + "..." if len(first_msg) > 30 else first_msg
        
        sessions.append({
            "session_id": doc["session_id"],
            "title": title
        })
        
    return sessions

@app.get("/chat/{session_id}")
def get_chat_history(session_id: str, user_email: str = Depends(verify_token)):
    """Returns the full message history and sources for a specific session."""
    if chats_collection is None:
        return {"messages": []}
        
    # Ensure they only fetch their own session
    doc = chats_collection.find_one({"session_id": session_id, "user_email": user_email}, {"_id": 0})
    if not doc:
        return {"messages": []}
        
    return {"messages": doc.get("messages", [])}


@app.get("/documents")
def list_documents(user_email: str = Depends(verify_token)):
    """Lists all uploaded documents for the user."""
    docs_list = []
    user_docs_path = os.path.join("docs", user_email)
    
    if os.path.exists(user_docs_path):
        for filename in os.listdir(user_docs_path):
            filepath = os.path.join(user_docs_path, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                tokens = size // 4
                cost = (tokens / 1000000) * 0.10
                docs_list.append({
                    "filename": filename,
                    "tokens": tokens,
                    "cost": round(cost, 6)
                })
    return docs_list

@app.delete("/documents/{filename}")
def delete_document(filename: str, user_email: str = Depends(verify_token)):
    """Deletes a document from the user's isolated folder and purges their Pinecone vectors."""
    filepath = os.path.join("docs", user_email, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX"))
        # Using multi-tenant deletion metadata
        index.delete(filter={"user": user_email, "source": filepath})
        print(f"Successfully purged Pinecone vectors for {filepath} under {user_email}")
    except Exception as e:
        print(f"Failed to delete vectors from Pinecone: {e}")
        
    os.remove(filepath)
    return {"status": "success"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), user_email: str = Depends(verify_token)):
    """
    Accepts a document upload, saves it into the isolated user folder,
    and re-indexes it into Pinecone with the user metadata.
    """
    allowed_types = [".pdf", ".txt", ".md", ".csv"]
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        user_docs_path = os.path.join("docs", user_email)
        os.makedirs(user_docs_path, exist_ok=True)
        
        save_path = os.path.join(user_docs_path, file.filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        from ingest import ingest_single_file
        # Notice we are passing user_email to securely tag the vectors
        chunks = ingest_single_file(save_path, user_email)
        time.sleep(5)

        return {
            "message": "Uploaded successfully.",
            "chunks": chunks,
            "file": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))