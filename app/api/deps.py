"""FastAPI dependencies."""

from functools import lru_cache

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.rag_service import RAGService, get_rag_service


@lru_cache
def _cached_settings() -> Settings:
    return get_settings()


def settings_dep() -> Settings:
    return _cached_settings()


def rag_service_dep(settings: Settings = Depends(settings_dep)) -> RAGService:
    return get_rag_service(settings)
