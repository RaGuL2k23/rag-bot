from sqlalchemy import create_engine, Column, String, Text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker 
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql://postgres:password@localhost:5432/chatbot"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Table definition
class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(String, primary_key=True)
    session_id = Column(String, index=True)
    role = Column(String)        # "user" or "assistant"
    content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
# Create tables if they don't exist
Base.metadata.create_all(engine)