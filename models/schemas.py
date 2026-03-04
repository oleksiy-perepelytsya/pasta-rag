import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class User(BaseModel):
    telegram_id: int
    username: Optional[str] = None
    first_name: str
    is_admin: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)


class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    telegram_id: int
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    vectorized: bool = False
    vectorized_at: Optional[datetime] = None
    message_count: int = 0


class Message(BaseModel):
    session_id: str
    telegram_id: int
    role: str  # "user" | "assistant"
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentConfig(BaseModel):
    key: str  # "system_prompt" | "llm_model"
    value: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: int  # telegram_id of admin


class Document(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    source_type: str  # "file" | "url" | "gdrive"
    source: str
    uploaded_by: int
    chunk_count: int = 0
    vectorized: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
