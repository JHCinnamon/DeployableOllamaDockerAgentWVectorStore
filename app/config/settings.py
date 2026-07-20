import logging
import os
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

CONFIG_FILE = Path(__file__).resolve()
APP_DIR = CONFIG_FILE.parents[1]
REPO_ROOT = CONFIG_FILE.parents[2]

# Load repo-level .env first, then app/.env to allow app-local overrides.
load_dotenv(dotenv_path=REPO_ROOT / ".env")
load_dotenv(dotenv_path=APP_DIR / ".env")


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3


class OpenAISettings(LLMSettings):
    """OpenAI-specific settings extending LLMSettings."""

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "ollama"))
    base_url: str = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    )
    default_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "llama3.2:3b")
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "nomic-embed-text")
    )


class DatabaseSettings(BaseModel):
    """Database connection settings."""

    service_url: str = Field(
        default_factory=lambda: os.getenv(
            "TIMESCALE_SERVICE_URL", "postgres://postgres:password@localhost:5432/postgres"
        )
    )


class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""

    table_name: str = Field(default_factory=lambda: os.getenv("VECTOR_TABLE_NAME", "embeddings"))
    embedding_dimensions: int = Field(
        default_factory=lambda: int(os.getenv("VECTOR_EMBEDDING_DIMENSIONS", "768"))
    )
    time_partition_interval: timedelta = timedelta(days=7)


class Settings(BaseModel):
    """Main settings class combining all sub-settings."""

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)


@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    settings = Settings()
    setup_logging()
    return settings
