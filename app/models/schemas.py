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


class RagEvaluationBlock(BaseModel):
    """LLM-as-judge scores persisted in ``evaluations`` (see ``app.services.evaluation``)."""

    evaluation_id: int
    relevance: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    notes: str | None = None


class SearchResponse(BaseModel):
    query: str
    answer: str
    source_chunks: list[SearchResultItem]
    evaluation: RagEvaluationBlock | None = None


class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_stored: int
