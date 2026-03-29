import hashlib

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB (creates a local folder called "chroma_db")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def ingest_pdf(pdf_path: str , session_id: str): 
    collection = chroma_client.get_or_create_collection(f"docs_{session_id}")
    # Step 1: extract text from PDF
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    # Step 2: split into chunks of 500 characters with 50 char overlap
    chunk_size = 500
    overlap = 100
    chunks = []
    i = 0
    while i < len(full_text):
        chunks.append(full_text[i:i+chunk_size])
        i += chunk_size - overlap

    # Step 3: convert chunks to vectors and store in ChromaDB
    for   chunk in (chunks):
        embedding = embedder.encode(chunk).tolist()
        # unique ID = hash of the chunk content itself
        chunk_id = hashlib.md5(chunk.encode()).hexdigest()
        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk]
        )
    print(f"✅ Ingested {len(chunks)} chunks from {pdf_path}")
    print("total in DB:", collection.count())


