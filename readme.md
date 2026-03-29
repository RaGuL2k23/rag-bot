# RAG Chatbot

## What it does
Upload any PDF and chat with it using AI. The bot remembers your conversation and answers questions based only on your document.

## Tech Stack
- **FastAPI** — REST API backend
- **ChromaDB** — vector database to store and search document chunks
- **Sentence Transformers** — converts text to vectors (all-MiniLM-L6-v2)
- **Redis** — fast in-memory conversation history (last 10 messages)
- **PostgreSQL** — permanent storage for all chat history
- **Groq LLM** — generates answers using llama-3.3-70b-versatile
- **Docker** — runs everything with one command

## How to Run
1. Clone the repo
2. Create a `.env` file:
   ```
   GROQ_API_KEY=your_key_here
   REDIS_URL=redis://redis:6379
   DATABASE_URL=postgresql://postgres:password@postgres:5432/chatbot
   ```
3. Add your PDF to the project folder and run:
   ```bash
   python ingest.py
   ```
4. Start everything:
   ```bash
   docker-compose up --build
   ```
5. Go to `http://localhost:8000/docs`

## Architecture

```
User Question
     ↓
FastAPI /chat endpoint
     ↓
1. Redis  →  fetch last 10 messages (fast memory)
     ↓
2. ChromaDB  →  convert question to vector → find top 3 similar chunks from PDF
     ↓
3. Groq LLM  →  answer using chunks + conversation history
     ↓
4. Redis  →  save new message
5. PostgreSQL  →  save permanently
     ↓
Response to User
```

**Ingest Flow (run once per PDF):**
```
PDF → extract text → split into 500 char chunks → 
convert to vectors → store in ChromaDB
```

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest/{session_id}` | Upload and ingest a PDF for a session |
| POST | `/rag-chat` | Send a message, get AI response | 
| GET | `/history/{session_id}` | Get full chat history  (function exist api should be done)
| DELETE | `/clear-history/{session_id}` | Clear session history |
