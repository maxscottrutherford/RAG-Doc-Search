"""RAG retrieval and generation orchestration (implement against your DB schema)."""

from app.core.config import Settings
from app.models.schemas import SearchRequest, SearchResponse, SearchResultItem


class RAGService:
    """Coordinates embedding, vector search (pgvector), and optional LLM answering."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def search(self, request: SearchRequest) -> SearchResponse:
        """
        Placeholder: embed `request.query`, query pgvector for nearest chunks,
        then optionally call the LLM with retrieved context.
        """
        _ = self._settings
        return SearchResponse(query=request.query, results=[])


def get_rag_service(settings: Settings) -> RAGService:
    return RAGService(settings)
