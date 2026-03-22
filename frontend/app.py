import streamlit as st
import requests
import os

# ─── Config ─────────────────────────────────────────────────────────
API_URL = "http://localhost:8000/chat"

st.set_page_config(
    page_title="RAG Knowledge Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ─── Header ─────────────────────────────────────────────────────────
st.title("🤖 RAG Knowledge Chatbot")
st.caption("Answers grounded in your documents — powered by Groq + Pinecone")
st.divider()

# ─── Initialize chat history in session state ───────────────────────
# session_state persists data across reruns in Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = {}

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
                # Send the entire history (excluding the current question which we just appended)
                history_to_send = st.session_state.messages[:-1]
                
                response = requests.post(
                    API_URL,
                    json={
                        "question": question,
                        "chat_history": history_to_send
                    },
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
    st.header("About")
    st.markdown("""
    This chatbot answers questions using **Retrieval Augmented Generation (RAG)**.

    **Stack:**
    - 🔍 Pinecone — vector search
    - 🧠 all-MiniLM-L6-v2 — embeddings
    - ⚡ Groq — LLM inference
    - 🔗 LangChain — RAG pipeline
    - 🚀 FastAPI — backend API
    - 🎈 Streamlit — this UI
    """)

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

    st.divider()

    # ─── Indexed documents list ──────────────────────────────────────
    st.header("Indexed documents")

    try:
        docs_list = os.listdir("docs")
        if docs_list:
            for doc in docs_list:
                st.caption(f"📄 {doc}")
        else:
            st.caption("No documents indexed yet.")
    except Exception:
        st.caption("Could not read docs folder.")

    st.divider()

    # ─── Clear chat button ───────────────────────────────────────────
    if st.button("🗑 Clear chat history"):
        st.session_state.messages = []
        st.session_state.sources = {}
        st.rerun()