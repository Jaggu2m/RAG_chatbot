import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# ─── Load environment variables from .env ───────────────────────────
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX")

# ─── Single file ingestion (safe to import by FastAPI) ───────────────
def ingest_single_file(file_path: str):
    """
    Ingests a single file into Pinecone.
    Called by FastAPI when a user uploads a document from the UI.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        docs = PyPDFLoader(file_path).load()
    elif ext in [".txt", ".md"]:
        docs = TextLoader(file_path, autodetect_encoding=True).load()
    elif ext == ".csv":
        docs = CSVLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    # Embed and upload
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY
    )

    return len(chunks)


# ─── Batch ingestion (only runs when executed directly) ──────────────
if __name__ == "__main__":
    print("\n Loading documents...")
    docs = []

    # Load PDFs
    for path in glob.glob("docs/**/*.pdf", recursive=True):
        print(f"  PDF: {path}")
        docs += PyPDFLoader(path).load()

    # Load plain text and markdown
    for path in glob.glob("docs/**/*.txt", recursive=True):
        print(f"  TXT: {path}")
        docs += TextLoader(path, autodetect_encoding=True).load()

    for path in glob.glob("docs/**/*.md", recursive=True):
        print(f"  MD : {path}")
        docs += TextLoader(path, autodetect_encoding=True).load()

    # Load CSVs (each row becomes a separate document)
    for path in glob.glob("docs/**/*.csv", recursive=True):
        print(f"  CSV: {path}")
        docs += CSVLoader(path).load()

    print(f"\n  Total pages/rows loaded: {len(docs)}")

    if len(docs) == 0:
        print("\n No documents found in /docs folder. Add some files and try again.")
        exit()

    # Chunk
    print("\n Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"  Total chunks created: {len(chunks)}")

    # Embed
    print("\n Loading embedding model (downloads once, then cached)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    print("  Embedding model ready")

    # Connect to Pinecone and create index if needed
    print("\n Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [i.name for i in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"  Index '{INDEX_NAME}' not found — creating it now...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"  Index '{INDEX_NAME}' created successfully")
    else:
        print(f"  Index '{INDEX_NAME}' already exists — skipping creation")

    # Embed and upload
    print(f"\n Embedding and uploading {len(chunks)} chunks to Pinecone...")
    print("  This may take a minute depending on document size...\n")

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY
    )

    print("\n All done! Your documents are indexed and ready to search.")
    print(f"  Index name      : {INDEX_NAME}")
    print(f"  Chunks uploaded : {len(chunks)}")
    print(f"  Embedding model : all-MiniLM-L6-v2 (384 dimensions)")
    print(f"  Vector store    : Pinecone cloud (us-east-1)\n")