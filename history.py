import os

import redis 
import json
from database import SessionLocal, ChatHistory
import uuid

MAX_HISTORY = 30  # keep last 10 messages
r = redis.from_url(os.getenv("REDIS_URL") or "redis://localhost:6379", decode_responses=True)

def get_history_redis(session_id: str) -> list :
    history = r.get(f"chat:{session_id}")
    print(f"Retrieved history for session {session_id}: {history}")
    return json.loads(str(history)) if history else []

def save_history_redis(session_id: str, messages: list):
    # keep only last MAX_HISTORY messages
    trimmed = messages[-MAX_HISTORY:]
    r.set(f"chat:{session_id}", json.dumps(trimmed))

def clear_redis_history(session_id: str):
    r.delete(f"chat:{session_id}")




def get_history(session_id: str) -> list:
    db = SessionLocal()
    row = db.query(ChatHistory).filter(ChatHistory.session_id == session_id).order_by(ChatHistory.created_at).limit(10).all()
    history = [{"role": record.role, "content": record.content} for record in row]
    db.close()
    return history
def save_history(session_id: str, role: str, content: str):
    db = SessionLocal()
    msg = ChatHistory(id =  str(uuid.uuid4()), session_id=session_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.close()