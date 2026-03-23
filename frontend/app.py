import streamlit as st
import requests
import os
import uuid
import jwt

# ─── Config ─────────────────────────────────────────────────────────
API_URL = "http://localhost:8000/chat"
SESSIONS_URL = "http://localhost:8000/sessions"
CHAT_HISTORY_URL = "http://localhost:8000/chat/{session_id}"

st.set_page_config(
    page_title="RAG Knowledge Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ─── Auth Parsing ───────────────────────────────────────────────────
if "token" in st.query_params:
    st.session_state.token = st.query_params.get("token")
    st.query_params.clear()

if "token" not in st.session_state:
    st.title("🔐 RAG Knowledge Base")
    st.markdown("Please log in with your Google account to access your private chatbot and isolated documents.")
    st.markdown(
        f'<a href="http://localhost:8000/auth/login" target="_self">'
        f'<button style="background-color:#4285F4;color:white;padding:10px 24px;border:none;border-radius:4px;cursor:pointer;font-size:16px;font-weight:bold;">'
        f'Sign in with Google'
        f'</button></a>',
        unsafe_allow_html=True
    )
    st.stop()

# Parse token (we don't verify signature here since backend does it, we just trust it to extract email)
try:
    payload = jwt.decode(st.session_state.token, options={"verify_signature": False})
    user_email = payload.get("sub", "Unknown User")
except Exception:
    st.error("Session expired or invalid token. Please log in again.")
    if "token" in st.session_state:
        del st.session_state.token
    st.stop()

# Helper for authenticated requests
def get_auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}

# ─── Header ─────────────────────────────────────────────────────────
st.title("🤖 RAG Knowledge Chatbot")
st.caption("Answers grounded in your documents — powered by Groq + Pinecone")
st.divider()

# ─── Fetch chat sessions from backend ───────────────────────────────
def fetch_sessions():
    try:
        response = requests.get(SESSIONS_URL, headers=get_auth_headers(), timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []

# ─── Fetch user documents ───────────────────────────────────────────
def fetch_user_documents():
    try:
        response = requests.get("http://localhost:8000/documents", headers=get_auth_headers(), timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []

# ─── Load a specific chat session ───────────────────────────────────
def load_session(session_id):
    try:
        response = requests.get(CHAT_HISTORY_URL.format(session_id=session_id), headers=get_auth_headers(), timeout=5)
        if response.status_code == 200:
            st.session_state.messages = response.json().get("messages", [])
            st.session_state.current_session_id = session_id
    except Exception as e:
        st.error(f"Failed to load chat history: {e}")

# ─── Initialize chat history in session state ───────────────────────
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = {}

# If messages are empty but a session is active, try fetching
if not st.session_state.messages and st.session_state.current_session_id:
    load_session(st.session_state.current_session_id)

# ─── Display existing chat history ──────────────────────────────────
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources under bot messages
        if message["role"] == "assistant" and i in st.session_state.sources:
            sources = st.session_state.sources[i]
            if sources:
                with st.expander(f"📄 Sources ({len(sources)})"):
                    for source in sources:
                        st.caption(f"📁 {source['file']}")
                        st.code(source["content"], language=None)

# ─── Chat input ─────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about your documents..."):

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(question)

    # Save user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "session_id": st.session_state.current_session_id,
                        "question": question,
                        "target_documents": st.session_state.get("target_documents", [])
                    },
                    headers=get_auth_headers(),
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]

                    # Display the answer
                    st.markdown(answer)

                    # Display sources in expander
                    if sources:
                        with st.expander(f"📄 Sources ({len(sources)})"):
                            for source in sources:
                                st.caption(f"📁 {source['file']}")
                                st.code(source["content"], language=None)
                    else:
                        st.caption("No sources found for this answer.")

                    # Save to history
                    msg_index = len(st.session_state.messages)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                    st.session_state.sources[msg_index] = sources

                else:
                    error_msg = f"API error: {response.status_code}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

            except requests.exceptions.ConnectionError:
                msg = "Cannot connect to backend. Make sure FastAPI is running on port 8000."
                st.error(msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": msg
                })

            except requests.exceptions.Timeout:
                msg = "Request timed out. Please try again."
                st.error(msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": msg
                })

# ─── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.caption(f"👤 Logged in as: **{user_email}**")
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

    st.divider()

    st.header("🎯 Target Documents")
    user_docs = fetch_user_documents()
    doc_filenames = [d["filename"] for d in user_docs]
    
    st.session_state.target_documents = st.multiselect(
        "Focus search on (leave blank for all):",
        options=doc_filenames,
        default=st.session_state.get("target_documents", []),
        help="If you select specific files here, Groq will completely ignore all other files."
    )

    st.divider()

    st.header("💬 Chat Sessions")
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.sources = {}
        st.rerun()
        
    sessions = fetch_sessions()
    if sessions:
        st.markdown("**Previous Conversations:**")
        for session in sessions:
            # Highlight current session
            btn_label = f"🟢 {session['title']}" if session['session_id'] == st.session_state.current_session_id else f"📄 {session['title']}"
            if st.button(btn_label, key=f"session_{session['session_id']}", use_container_width=True):
                st.session_state.current_session_id = session['session_id']
                st.session_state.messages = []
                st.session_state.sources = {}
                load_session(session['session_id'])
                st.rerun()
    else:
        st.caption("No previous chats found.")

    st.divider()

    # ─── Document Upload Section ─────────────────────────────────────
    st.header("Upload a document")
    st.caption("Supported: PDF, TXT, MD, CSV")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "md", "csv"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if st.button("Upload & Index", type="primary"):
            with st.spinner(f"Uploading and indexing {uploaded_file.name}..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/upload",
                        files={"file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )},
                        headers=get_auth_headers(),
                        timeout=300  # 5 minutes specifically for large documents
                    )

                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Done! {data['chunks']} chunks indexed from '{data['file']}'")
                        st.caption("You can now ask questions about this document.")
                    else:
                        st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Make sure FastAPI is running.")
                except requests.exceptions.Timeout:
                    st.error("Upload timed out. Try a smaller file.")

    # ─── Manage Documents ────────────────────────────────────────────
    st.header("📂 Manage Documents")

    if user_docs:
        for doc in user_docs:
            with st.expander(f"📄 {doc['filename']}"):
                st.write(f"**Tokens:** ~{doc['tokens']:,}")
                st.write(f"**Est. Vector Cost:** ${doc['cost']:.6f}")
                if st.button("🗑️ Delete", key=f"del_{doc['filename']}", use_container_width=True):
                    with st.spinner("Purging vectors..."):
                        requests.delete(f"http://localhost:8000/documents/{doc['filename']}", headers=get_auth_headers(), timeout=60)
                        st.rerun()
    else:
        st.caption("No documents indexed yet.")

    st.divider()

    # ─── Clear chat button ───────────────────────────────────────────
    if st.button("🗑 Clear current chat screen"):
        st.session_state.messages = []
        st.session_state.sources = {}
        st.rerun()