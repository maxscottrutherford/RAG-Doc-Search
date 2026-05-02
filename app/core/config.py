"""Application settings loaded from environment variables."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the API and backing services."""

    app_name: str
    debug: bool
    database_url: str
    openai_api_key: str


def get_settings() -> Settings:
    database_url = os.getenv("DATABASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not database_url:
        raise ValueError("DATABASE_URL is required")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")
    return Settings(
        app_name=os.getenv("APP_NAME", "RAG Document Search API"),
        debug=os.getenv("DEBUG", "false").lower() in ("1", "true", "yes"),
        database_url=database_url,
        openai_api_key=openai_api_key,
    )
