"""
GCIS Backend — Core Configuration
All settings loaded from environment variables / .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── LLM Provider ─────────────────────────────────────────────────────────
    LLM_PROVIDER: str = "groq"                        # groq | gemini | ollama
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""                           # fallback
    ANTHROPIC_API_KEY: str = ""                        # reserved

    # ─── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://gcis:gcis_pass@localhost:5432/gcis_db"

    # ─── Redis ────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ─── Kafka ────────────────────────────────────────────────────────────────
    KAFKA_BROKERS: str = "localhost:9092"
    KAFKA_TOPIC_TAGS: str = "gcis.tags"
    KAFKA_TOPIC_FEATURES: str = "gcis.features"
    KAFKA_TOPIC_ALERTS: str = "gcis.alerts"
    KAFKA_TOPIC_EPISODES: str = "gcis.episodes"
    KAFKA_TOPIC_FEEDBACK: str = "gcis.feedback"

    # ─── Qdrant ───────────────────────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "gcis_episodes"
    QDRANT_VECTOR_SIZE: int = 128                     # match ML partner's embedding dim

    # ─── Agent Thresholds ─────────────────────────────────────────────────────
    ALERT_THRESHOLD: float = 0.65                     # p_offspec >= this → escalate
    MIN_CONFIDENCE: float = 0.40                      # confidence >= this to escalate
    OPERATOR_AWAIT_TIMEOUT_SEC: int = 180             # 3-minute operator timeout
    STALE_TAG_FAST_SEC: int = 10                      # fast loop stale threshold
    STALE_TAG_SLOW_SEC: int = 60                      # slow loop stale threshold
    ALERT_DEDUP_WINDOW_SEC: int = 30                  # deduplication window

    # ─── Backend ──────────────────────────────────────────────────────────────
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    SECRET_KEY: str = "change_me_before_production"

    # ─── Frontend ─────────────────────────────────────────────────────────────
    NEXT_PUBLIC_API_BASE: str = "http://localhost:8000"
    NEXT_PUBLIC_WS_BASE: str = "ws://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
