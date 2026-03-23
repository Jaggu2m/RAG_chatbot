import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest

# ─── Load environment variables ─────────────────────────────────────
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")

# ─── Step 1: Load embedding model ───────────────────────────────────
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# ─── Step 2: Connect to Pinecone ────────────────────────────────────
print("Connecting to Pinecone...")
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)

# ─── Load Re-ranker ─────────────────────────────────────────────────
print("Loading Re-ranker model...")
# Using lightweight default cross-encoder for re-ranking
ranker = Ranker(cache_dir=".")

# ─── Step 3: Set up retriever as a function (refreshes on each call) ─
def get_retriever(user_email: str, target_documents: list = None):
    """
    Returns a fresh retriever every time.
    We fetch 10 documents directly from Pinecone. The Re-ranker will filter these down to 3.
    Critically, we apply a strict metadata filter so users can ONLY retrieve their own uploaded files.
    """
    filter_dict = {"user": user_email}
    
    # Mathematical document exclusion: If target files are provided, ignore the rest
    if target_documents:
        sources = [os.path.join("docs", user_email, doc) for doc in target_documents]
        filter_dict["source"] = {"$in": sources}
        
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 15, "filter": filter_dict}
    )

# ─── Step 4: Prompt template ────────────────────────────────────────
prompt = PromptTemplate.from_template("""
You are a highly analytical expert assistant that answers questions based ONLY on the 
context provided below. You are excellent at understanding complex topics.

If the answer is not found in the context, say exactly:
"I don't have information about that in the knowledge base."

Your instructions for answering:
1. Provide highly detailed, comprehensive explanations.
2. Form your reply into clearly readable paragraphs, and use bullet points when explaining lists or multiple steps.
3. If the user asks for a detailed explanation, thoroughly unpack and explain all the relevant concepts found in the context. Do not be overly brief.
4. Always cite precisely which document or source file your answer comes from.

Context:
{context}

Question:
{question}

Answer:
""")

# ─── Step 5: Groq LLM ───────────────────────────────────────────────
print("Connecting to Groq...")
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=2048,
)

# ─── Step 6: Helper to format retrieved docs ────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

print("RAG pipeline ready\n")

# ─── Step 7: Main function called by FastAPI ────────────────────────
def ask_question(question: str, user_email: str, target_documents: list = None, chat_history: list = None) -> dict:
    if not question.strip():
        return {
            "answer": "Please ask a valid question.",
            "sources": []
        }

    retriever = get_retriever(user_email, target_documents)
    
    # Get top 10 rough documents from Pinecone
    rough_docs = retriever.invoke(question)
    
    # Apply FlashRank Cross-Encoder Re-ranking
    source_docs = []
    if rough_docs:
        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(rough_docs)
        ]
        rerankrequest = RerankRequest(query=question, passages=passages)
        results = ranker.rerank(rerankrequest)
        
        # Take the top 7 highly contextual results to maximize detail
        top_results = results[:7]
        
        for p in top_results:
            source_docs.append(Document(page_content=p["text"], metadata=p.get("meta", {})))
    
    # Manually invoke the LLM with the highly contextual top 3 results
    context_text = format_docs(source_docs)
    prompt_value = prompt.format(context=context_text, question=question)
    
    response = llm.invoke(prompt_value)
    answer = response.content

    # Extract sources
    sources = []
    for doc in source_docs:
        source = {
            "file": doc.metadata.get("source", "Unknown"),
            "content": doc.page_content[:200]
        }
        if source not in sources:
            sources.append(source)

    return {
        "answer": answer,
        "sources": sources
    }

# ─── Quick test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    test_questions = [
        "What is the return policy?",
        "Do you offer free shipping?",
    ]

    for q in test_questions:
        print(f"Q: {q}")
        result = ask_question(q)
        print(f"A: {result['answer']}")
        print(f"Sources: {[s['file'] for s in result['sources']]}")
        print()