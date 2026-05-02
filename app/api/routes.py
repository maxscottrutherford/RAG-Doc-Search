"""API route definitions."""

from fastapi import APIRouter, Depends

from app.api.deps import rag_service_dep, settings_dep
from app.core.config import Settings
from app.models.schemas import HealthResponse, SearchRequest, SearchResponse
from app.services.rag_service import RAGService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(settings_dep)) -> HealthResponse:
    _ = settings
    return HealthResponse()


@router.post("/search", response_model=SearchResponse)
def search(
    body: SearchRequest,
    rag: RAGService = Depends(rag_service_dep),
) -> SearchResponse:
    return rag.search(body)
