"""Request and response models for the HTTP API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve")


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
