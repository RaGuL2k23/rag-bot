# Run this ONCE to load your PDF into ChromaDB

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB (creates a local folder called "chroma_db")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("documents")

def ingest_pdf(pdf_path: str):
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
    for idx, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        collection.add(
            ids=[f"chunk_{idx}"],
            embeddings=[embedding],
            documents=[chunk]
        )
    print(f"✅ Ingested {len(chunks)} chunks from {pdf_path}")
    print("total in DB:", collection.count())


if __name__ == "__main__":
    ingest_pdf("10TH MARKSHEET.pdf")  # <-- change to your PDF path