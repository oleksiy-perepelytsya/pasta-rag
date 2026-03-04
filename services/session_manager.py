import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from config import settings
from models.schemas import Message, Session
from storage.chroma import ChromaStore
from storage.mongo import MongoStore

logger = logging.getLogger(__name__)


def _format_rag_block(past_sessions: list[dict], kb_hits: list[dict]) -> str:
    parts = []
    if past_sessions:
        parts.append("=== Relevant past conversations ===")
        for hit in past_sessions:
            parts.append(hit.get("content", ""))
    if kb_hits:
        parts.append("=== Relevant knowledge base ===")
        for hit in kb_hits:
            meta = hit.get("metadata", {})
            source = meta.get("filename", "")
            if source:
                parts.append(f"[Source: {source}]")
            parts.append(hit.get("content", ""))
    return "\n\n".join(parts)


class SessionManager:
    def __init__(self, mongo: MongoStore, chroma: ChromaStore):
        self.mongo = mongo
        self.chroma = chroma

    async def get_or_create_session(self, telegram_id: int) -> Session:
        session = await self.mongo.get_active_session(telegram_id)
        if session is None:
            session = Session(telegram_id=telegram_id)
            session = await self.mongo.create_session(session)
            logger.info(f"New session created: {session.session_id} for user {telegram_id}")
        return session

    async def touch_session(self, session: Session) -> None:
        session.last_active = datetime.utcnow()
        session.message_count += 1
        await self.mongo.update_session(session)

    async def build_context(
        self,
        telegram_id: int,
        current_message: str,
        session: Session,
    ) -> tuple[str, list[dict]]:
        """
        Returns (system_prompt_with_rag, current_session_messages_list).
        The caller should append the current user message before calling LLM.
        """
        # 1. System prompt from DB
        system_prompt = (
            await self.mongo.get_config("system_prompt")
            or "You are a helpful assistant."
        )

        # 2. RAG: past sessions (per-user ChromaDB namespace)
        past_sessions = await self.chroma.query_sessions(
            telegram_id, current_message, settings.max_rag_results
        )

        # 3. RAG: shared knowledge base
        kb_hits = await self.chroma.query_knowledge_base(
            current_message, settings.max_rag_results
        )

        # 4. Inject RAG block into system prompt
        rag_block = _format_rag_block(past_sessions, kb_hits)
        if rag_block:
            system_prompt = system_prompt + "\n\n" + rag_block

        # 5. Current session messages from MongoDB
        recent_msgs = await self.mongo.get_session_messages(
            session.session_id, settings.max_context_messages
        )
        messages = [{"role": m.role, "content": m.content} for m in recent_msgs]

        return system_prompt, messages

    async def vectorize_session(self, session: Session) -> None:
        messages = await self.mongo.get_session_messages(session.session_id, limit=9999)
        if not messages:
            await self.mongo.mark_session_vectorized(session.session_id)
            return

        text = "\n".join(
            f"{m.role.upper()}: {m.content}" for m in messages
        )
        metadata = {
            "session_id": session.session_id,
            "telegram_id": session.telegram_id,
            "started_at": session.started_at.isoformat(),
            "message_count": session.message_count,
        }
        await self.chroma.add_session(
            session.telegram_id, session.session_id, text, metadata
        )
        await self.mongo.mark_session_vectorized(session.session_id)
        logger.info(f"Session {session.session_id} vectorized successfully")


async def vectorize_stale_sessions_loop(
    mongo: MongoStore, chroma: ChromaStore
) -> None:
    """Background asyncio task — runs forever until cancelled."""
    manager = SessionManager(mongo, chroma)
    timeout = timedelta(minutes=settings.session_timeout_minutes)
    logger.info(
        f"Vectorization loop started (timeout={settings.session_timeout_minutes}m)"
    )
    while True:
        try:
            await asyncio.sleep(60)
            cutoff = datetime.utcnow() - timeout
            stale = await mongo.get_stale_sessions(cutoff)
            if stale:
                logger.info(f"Vectorizing {len(stale)} stale session(s)")
            for session in stale:
                try:
                    await manager.vectorize_session(session)
                except Exception as e:
                    logger.error(
                        f"Failed to vectorize session {session.session_id}: {e}",
                        exc_info=True,
                    )
        except asyncio.CancelledError:
            logger.info("Vectorization loop cancelled")
            break
        except Exception as e:
            logger.error(f"Vectorization loop error: {e}", exc_info=True)
