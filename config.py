from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Telegram
    telegram_bot_token: str = Field(...)

    # LLM API keys
    gemini_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    default_llm_model: str = Field(default="gemini-2.0-flash")

    # MongoDB
    mongodb_uri: str = Field(default="mongodb://localhost:27017")
    mongodb_database: str = Field(default="pasta_rag")

    # ChromaDB
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8000)

    # Session behaviour
    session_timeout_minutes: int = Field(default=30)
    max_context_messages: int = Field(default=50)
    max_rag_results: int = Field(default=3)

    # Logging
    log_level: str = Field(default="INFO")


settings = Settings()
