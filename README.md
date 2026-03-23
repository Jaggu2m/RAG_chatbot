# Secure Multi-Tenant RAG AI Platform 🤖

This project is a highly advanced, intelligent document chatbot. It allows multiple users to securely log in via Google OAuth, upload their private documents (PDF, CSV, MD, TXT), and ask complex questions against those specific documents using state-of-the-art AI. Every answer provides precise source citations.

---

## 🏗️ The Architecture

The system operates on a dual-pipeline Multi-Tenant architecture to guarantee absolute privacy and accuracy for each user login:

### 1. The Ingestion Pipeline (When a user clicks "Upload")
- The user uploads a file through the **Streamlit UI**. The **FastAPI Backend** intercepts the file along with the user's `JWT Token` mapping to their email address.
- **LlamaParse** breaks down complex PDFs natively converting tables and images accurately into Markdown, while **LangChain Loaders** ingest CSVs and plain text.
- The raw text is broken down into structured overlapping chunks, converted into a high-dimensional vector using a local embedding model (`all-MiniLM-L6-v2`), and forcefully tagged with `"user": "<email>"` metadata.
- These secured vectors are then permanently cached in **Pinecone Cloud**.

### 2. The Retrieval Query Pipeline (When a user types a question)
- The user asks a question in the UI. The backend authenticates them via the `Authorization: Bearer <token>`.
- The question is embedded into a standard vector. Pinecone is queried for the **Top 10** most mathematically similar document chunks restricted *strictly* to that specific user's namespace metadata.
- **FlashRank Re-Ranking:** A secondary active Cross-Encoder (`ms-marco-TinyBERT`) takes those 10 raw documents and forces them to actively compete against the exact user query. It slices the 10 documents down to the absolute best **Top 3 purely contextual matches**.
- The highly relevant documents are patched into a system instruction Prompt and handed off to **Groq's API running Llama 3.3 (70B)** to deliver a lightning-fast contextual conversational reply.
- The conversation gets logged indefinitely into **MongoDB Atlas**.

---

## 🛠️ Technology Stack & Cost Breakdown

| Tool | Role | Cost |
| :--- | :--- | :--- |
| **FastAPI** / **Python** | Backend Server / API | ✅ Completely free |
| **Streamlit** | Frontend Web UI | ✅ Completely free |
| **Authlib** / **Google Cloud** | Multi-Tenant User Authentication | ✅ Free |
| **MongoDB Atlas** | Persistent Chat History & Sessions Storage | ✅ Free Tier usage |
| **pinecone** | Vector Storage & Mathematical Searching | ✅ Free Starter Tier |
| **all-MiniLM-L6-v2** | Transforms Text to Vector numerical mappings | ✅ Completely free (runs locally) |
| **FlashRank (TinyBERT)** | Advanced Document Re-ranking for Extreme Accuracy | ✅ Completely free (runs locally) |
| **LlamaParse (LlamaCloud)**| Enterprise-Grade Visual PDF extraction & markdown processing | ✅ Free Tier (1,000 pages per day) |
| **Groq (Llama 3.3 70B)** | Blistering-fast Large Language Model Generation | ⚠️ Generous Free Trial / Exceptionally cheap API usage |
| **Langchain** | Orchestration framework linking models, parsers, and loaders | ✅ Completely free | 

---

## 🚀 How to Host & Run Locally

Follow these instructions perfectly to deploy the two-stack platform locally on your computer.

### Step 1: Clone and Set Up Python
Ensure you have Python installed.
```bash
# Clone your private codebase
git clone [repository_url]
cd rag-chatbot

# Create an isolated python environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate
# Activate it (Mac/Linux)
source venv/bin/activate

# Install all architecture dependencies
pip install -r requirements.txt
```

### Step 2: Set Environment Variables (`.env`)
Create a file named `.env` in the root of the project with the following structure (Replace placeholder text with actual keys generated from Provider websites):

```env
# Vector Database
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=rag-knowledge-base

# Generative AI Key
GROQ_API_KEY=your_groq_api_key

# Document Parser AI
LLAMA_CLOUD_API_KEY=your_llamacloud_api_key

# Chat Session Storage
MONGO_URI=mongodb+srv://admin:<password>@cluster0.xxxxx.mongodb.net/

# Google OAuth Keys (Cloud Console)
GOOGLE_CLIENT_ID=your-google-oauth-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-oauth-client-secret

# System Cryptography Token Key (Random long string)
JWT_SECRET=ultra-super-secret-password-string
```

### Step 3: Boot The Application
The application inherently operates across two independent servers. 

**Terminal 1 for the Backend API**:
```bash
# Activate your environment
venv\Scripts\activate
# Start the FastAPI engine on port 8000
uvicorn backend.main:app --reload --port 8000
```
*Notice: `application startup complete` means the models successfully downloaded to your computer and connected to Pinecone databases.*

**Terminal 2 for the Frontend Visual Interface**:
```bash
# Open a completely new terminal and go to the project root
venv\Scripts\activate
# Launch the Streamlit viewer
streamlit run frontend/app.py
```

### Step 4: Access The Platform!
Streamlit will automatically bind to `localhost:8501` and launch in your default web browser.

Click **"Sign in with Google"**. Since you set `http://localhost:8000/auth/callback` inside your Google API Auth Console, Google will verify you, securely pass a cryptographic JWT token to the backend, and effortlessly route you back to the core Chatbot UI! Upload a PDF on the side, view its estimated extraction token cost, ask a question on the main screen, and witness the system securely invoke sources!

--- 
*Note: To purge storage to 0 across the entire architecture, users can easily click the red "Delete" button inside the **Manage Documents** sidebar UI block which reaches across the network out to Pinecone servers and local `/docs/user` system folders specifically tied to their accounts simultaneously, deleting their data safely!*
