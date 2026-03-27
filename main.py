import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import chromadb 
from sentence_transformers import SentenceTransformer
from history import get_history_redis, save_history_redis , clear_redis_history , get_history, save_history

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("documents")
# print(collection.count())  # add this to check if collection is accessible
class ChatRequest(BaseModel):
    message: str
    session_id: str


@app.post("/rag-chat")
async def chat(request: ChatRequest):
    session_id = request.session_id 
    history = get_history_redis(session_id)  
    if not history:  # if no history in Redis, load from Postgres
        history = get_history(session_id)

    # Step 1: convert question to vector
    question_vector = embedder.encode(request.message).tolist()

    history.append({"role": "user", "content": request.message})  # add user message to history
    save_history(session_id, "user", request.message)  
    
    # Step 2: find top 3 most similar chunks from ChromaDB
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=3
    ) 
    relevant_chunks = "\n\n".join((results["documents"] or[""])[0]) 
    print(f"Relevant chunks for session {session_id}: {relevant_chunks}")  # debug print
    # Step 3: send question + relevant chunks to LLM
    response = client.chat.completions.create(
        # model="llama-3.3-70b-versatile",
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": f"Answer the user's question using only the context below:\n\n{relevant_chunks} \n\n Here is the conversation history:\n\n{history}"},
            {"role":"system" , "content": "You are a helpful assistant. be a nice friend to the user."},  
        ]
    )
    history.append({"role": "assistant", "content": response.choices[0].message.content})  # add assistant message to history
    save_history(session_id, "assistant", response.choices[0].message.content or "")
    save_history_redis(session_id, history)  # <-- save the updated history
    return {"response": response.choices[0].message.content}

@app.delete("/clear-history/{session_id}")
async def clear_history(session_id: str):
    clear_redis_history(session_id)
    return {"message": "History cleared successfully"}