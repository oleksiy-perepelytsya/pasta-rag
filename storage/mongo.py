import logging
from datetime import datetime
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING, TEXT, ReturnDocument

from models.schemas import AgentConfig, Document, Message, Session, User

logger = logging.getLogger(__name__)


class MongoStore:
    def __init__(self, uri: str, db_name: str):
        self.client = AsyncIOMotorClient(uri)
        self.db = self.client[db_name]
        self.users = self.db.users
        self.sessions = self.db.sessions
        self.messages = self.db.messages
        self.agent_config = self.db.agent_config
        self.documents = self.db.documents

    async def init_db(self) -> None:
        await self.client.admin.command("ping")
        logger.info("MongoDB connected")
        await self._create_indexes()

    async def _create_indexes(self) -> None:
        await self.users.create_index("telegram_id", unique=True)
        await self.users.create_index("username")
        await self.sessions.create_index(
            [("telegram_id", ASCENDING), ("vectorized", ASCENDING)]
        )
        await self.sessions.create_index("last_active")
        await self.messages.create_index(
            [("session_id", ASCENDING), ("created_at", ASCENDING)]
        )
        await self.messages.create_index([("content", TEXT)])
        await self.agent_config.create_index("key", unique=True)
        await self.documents.create_index("doc_id", unique=True)
        logger.info("MongoDB indexes created")

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    async def upsert_user(self, user: User) -> User:
        doc = await self.users.find_one_and_update(
            {"telegram_id": user.telegram_id},
            {
                "$set": {
                    "username": user.username,
                    "first_name": user.first_name,
                    "last_active": user.last_active,
                },
                "$setOnInsert": {
                    "is_admin": user.is_admin,
                    "created_at": user.created_at,
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        doc.pop("_id", None)
        return User(**doc)

    async def get_user(self, telegram_id: int) -> Optional[User]:
        doc = await self.users.find_one({"telegram_id": telegram_id})
        if doc:
            doc.pop("_id", None)
            return User(**doc)
        return None

    async def get_user_by_username(self, username: str) -> Optional[User]:
        doc = await self.users.find_one({"username": username})
        if doc:
            doc.pop("_id", None)
            return User(**doc)
        return None

    async def count_users(self) -> int:
        return await self.users.count_documents({})

    async def promote_user(self, telegram_id: int) -> bool:
        result = await self.users.update_one(
            {"telegram_id": telegram_id},
            {"$set": {"is_admin": True}},
        )
        return result.matched_count > 0

    async def get_all_admins(self) -> list[User]:
        admins = []
        async for doc in self.users.find({"is_admin": True}):
            doc.pop("_id", None)
            admins.append(User(**doc))
        return admins

    async def get_all_users(self, limit: int = 100) -> list[User]:
        users = []
        async for doc in self.users.find().sort("created_at", DESCENDING).limit(limit):
            doc.pop("_id", None)
            users.append(User(**doc))
        return users

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def create_session(self, session: Session) -> Session:
        await self.sessions.insert_one(session.model_dump())
        return session

    async def get_active_session(self, telegram_id: int) -> Optional[Session]:
        doc = await self.sessions.find_one(
            {"telegram_id": telegram_id, "vectorized": False},
            sort=[("started_at", DESCENDING)],
        )
        if doc:
            doc.pop("_id", None)
            return Session(**doc)
        return None

    async def update_session(self, session: Session) -> None:
        await self.sessions.update_one(
            {"session_id": session.session_id},
            {
                "$set": {
                    "last_active": session.last_active,
                    "message_count": session.message_count,
                }
            },
        )

    async def get_stale_sessions(self, cutoff: datetime) -> list[Session]:
        docs = await self.sessions.find(
            {"last_active": {"$lt": cutoff}, "vectorized": False}
        ).to_list(None)
        sessions = []
        for doc in docs:
            doc.pop("_id", None)
            sessions.append(Session(**doc))
        return sessions

    async def mark_session_vectorized(self, session_id: str) -> None:
        await self.sessions.update_one(
            {"session_id": session_id},
            {"$set": {"vectorized": True, "vectorized_at": datetime.utcnow()}},
        )

    async def count_sessions(self) -> int:
        return await self.sessions.count_documents({})

    async def count_vectorized_sessions(self) -> int:
        return await self.sessions.count_documents({"vectorized": True})

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    async def save_message(self, message: Message) -> Message:
        await self.messages.insert_one(message.model_dump())
        return message

    async def get_session_messages(self, session_id: str, limit: int = 50) -> list[Message]:
        cursor = (
            self.messages.find({"session_id": session_id})
            .sort("created_at", ASCENDING)
        )
        docs = await cursor.to_list(None)
        msgs = []
        for doc in docs:
            doc.pop("_id", None)
            msgs.append(Message(**doc))
        # Return last `limit` messages in chronological order
        return msgs[-limit:] if len(msgs) > limit else msgs

    async def count_messages(self) -> int:
        return await self.messages.count_documents({})

    async def get_user_messages(self, telegram_id: int, limit: int = 200) -> list[Message]:
        cursor = (
            self.messages.find({"telegram_id": telegram_id})
            .sort("created_at", ASCENDING)
            .limit(limit)
        )
        msgs = []
        async for doc in cursor:
            doc.pop("_id", None)
            msgs.append(Message(**doc))
        return msgs

    async def search_messages(self, query: str, limit: int = 5) -> list[Message]:
        cursor = self.messages.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}},
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        msgs = []
        async for doc in cursor:
            doc.pop("_id", None)
            doc.pop("score", None)
            msgs.append(Message(**doc))
        return msgs

    # ------------------------------------------------------------------
    # Agent config
    # ------------------------------------------------------------------

    async def get_config(self, key: str) -> Optional[str]:
        doc = await self.agent_config.find_one({"key": key})
        return doc["value"] if doc else None

    async def set_config(self, key: str, value: str, admin_id: int) -> None:
        cfg = AgentConfig(key=key, value=value, updated_by=admin_id)
        await self.agent_config.find_one_and_update(
            {"key": key},
            {"$set": cfg.model_dump()},
            upsert=True,
        )

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    async def save_document(self, doc: Document) -> Document:
        await self.documents.insert_one(doc.model_dump())
        return doc

    async def update_document_vectorized(self, doc_id: str, chunk_count: int) -> None:
        await self.documents.update_one(
            {"doc_id": doc_id},
            {"$set": {"vectorized": True, "chunk_count": chunk_count}},
        )

    async def count_documents(self) -> int:
        return await self.documents.count_documents({})

    async def get_recent_documents(self, limit: int = 10) -> list[Document]:
        docs = []
        async for doc in self.documents.find().sort("created_at", DESCENDING).limit(limit):
            doc.pop("_id", None)
            docs.append(Document(**doc))
        return docs
