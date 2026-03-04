# pasta-rag Project Memory

## Project Overview
Telegram RAG chatbot. Users chat freely; sessions stored in MongoDB, vectorized to ChromaDB on timeout. Admins manage prompt, model, docs, stats.

## Tech Stack
- python-telegram-bot 21.x (async polling)
- Motor 3.3.x + MongoDB (pasta_rag DB)
- ChromaDB (HttpClient, Gemini embeddings)
- Pydantic + pydantic-settings
- Multi-provider LLM: Gemini / Claude / OpenAI (auto-detected by model name prefix)

## File Structure
```
pasta-rag/
├── config.py                    # Pydantic BaseSettings from .env
├── models/schemas.py            # User, Session, Message, AgentConfig, Document
├── storage/mongo.py             # MongoStore (all CRUD)
├── storage/chroma.py            # ChromaStore (Gemini embeddings, per-user sessions + KB)
├── services/llm.py              # LLMClient (gemini-*/claude-*/gpt-* dispatch)
├── services/doc_loader.py       # PDF/DOCX/URL/GDrive extraction + chunking
├── services/session_manager.py  # SessionManager + vectorize_stale_sessions_loop
├── bot/keyboards.py             # Inline keyboard builders
├── bot/handlers/user.py         # UserHandlers (start, message)
├── bot/handlers/admin.py        # AdminHandlers (setprompt, setmodel, stats, search, upload, promote)
└── bot/main.py                  # Entry point, wires all components
```

## Key Patterns
- First user to chat becomes admin; others promoted via /promote
- Handler groups: group=0 for upload/state handlers, group=1 for regular chat
- State machine via context.user_data["state"]: AWAITING_UPLOAD / AWAITING_SETPROMPT / AWAITING_SETMODEL / AWAITING_SEARCH
- ChromaDB: always use Gemini embeddings (retrieval_document for add, retrieval_query for query) — never mix with ChromaDB default
- MongoDB: $setOnInsert for is_admin in upsert_user to preserve admin status
- Background task: vectorize_stale_sessions_loop (asyncio.create_task in main)
- ChromaDB collections: sessions_{telegram_id} (per-user), knowledge_base (shared)

## MongoDB Collections
- users, sessions, messages, agent_config, documents

## Run Commands
```bash
cp .env.example .env  # fill in tokens
pip install -r requirements.txt
python -m bot.main
# or with Docker:
docker-compose up -d
```
