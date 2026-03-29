import asyncio
import os 
import chromadb 
import tempfile

from dotenv import load_dotenv  
from groq import Groq
from sentence_transformers import SentenceTransformer
from fastapi import WebSocket
from history import get_history_redis, save_history_redis , clear_redis_history , get_history, save_history,clear_postgres_history
from fastapi import FastAPI, WebSocket, WebSocketDisconnect ,UploadFile, File
from ingest import ingest_pdf
from fastapi.middleware.cors import CORSMiddleware
# from ollama import chat as ollama_chat
 
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("documents")
# print(collection.count())  # add this to check if collection is accessible
 
@app.post("/ingest/{session_id}")
async def ingest_endpoint(session_id: str, file: UploadFile = File(...)):
    # save uploaded file to a temp file so fitz can open it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        ingest_pdf(tmp_path , session_id)  # pass session_id to ingest_pdf for session-specific handling
        return {"message": f"{file.filename} ingested successfully"}
    finally:
        os.remove(tmp_path)  # cleanup temp file after done


collection_cache = {}

def get_collection(session_id: str):
    if session_id not in collection_cache:
        collection_cache[session_id] = chroma_client.get_or_create_collection(f"docs_{session_id}")
    return collection_cache[session_id]

@app.websocket("/rag-chat")
async def chat(websocket: WebSocket):
    await websocket.accept()

    init_payload = await websocket.receive_json()
    session_id = init_payload.get("session_id", "default_session")

    history = get_history_redis(session_id)
    if not history:
        history = get_history(session_id)

    try:
        while True:
            if not init_payload:
                payload = await websocket.receive_json()
            else:
                payload = init_payload
                init_payload = None
            user_message = payload.get("message", "")

            if not user_message.strip():
                continue

            # non-blocking embed
            question_vector = await asyncio.get_event_loop().run_in_executor(
                None, lambda: embedder.encode(user_message).tolist()
            )

            # cached collection lookup
            results = get_collection(session_id).query(
                query_embeddings=[question_vector],
                n_results=3
            )
            relevant_chunks = "\n\n".join((results["documents"] or [""])[0])

            print(f"Relevant chunks: {relevant_chunks}")
            messages: list = [
    {
        "role": "system",
        "content": (
            "You are a precise and helpful AI assistant.\n\n"

            "### PRIORITY RULES (strict):\n"
            "1. Answer ONLY the latest user question.\n"
            "2. Use provided CONTEXT if relevant.\n"
            "3. Use HISTORY only for continuity (do NOT repeat or summarize it).\n"
            "4. If context is insufficient, say 'I don't have enough information' instead of guessing.\n\n"

            "### CONTEXT:\n"
            f"{relevant_chunks}\n\n"

            "### HISTORY:\n"
            "Previous conversation is provided for reference.\n"
            "Do NOT repeat it unless explicitly asked.\n"
        )
    },
    *history,
    {
        "role": "user",
        "content": user_message
    }
]
            stream = client.chat.completions.create(
                # model="llama-3.3-70b-versatile",
                model="moonshotai/kimi-k2-instruct",
                messages=messages ,
                stream=True
            )
            
            # stream = ollama_chat  (
            #     model='qwen2.5:3b',
            #     messages=messages,
            #     stream=True
            # )

            final_response = []
            for chunk in stream:
                token = chunk.choices[0].delta.content
                # token = chunk.message.content

                if token:
                    await websocket.send_text(token)
                    final_response.append(token)

            await websocket.send_text("[DONE]")
            history.append({"role": "user", "content": user_message})
            save_history(session_id, "user", user_message)

            final_response_str = "".join(final_response)
            history.append({"role": "assistant", "content": final_response_str})
            save_history(session_id, "assistant", final_response_str or "")
            save_history_redis(session_id, history)

    except WebSocketDisconnect:
        print(f"Client disconnected — session: {session_id}")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.send_text(f"[ERROR] {str(e)}")
@app.delete("/clear-history/{session_id}")
async def clear_history(session_id: str):
    clear_redis_history(session_id)
    clear_postgres_history(session_id)
    return {"message": "History cleared successfully"}

@app.get("/history/{session_id}")
async def get_chat_history(session_id : str):     
    history = get_history_redis(session_id)
    if not history:
        history = get_history(session_id)
    return {"history": history}