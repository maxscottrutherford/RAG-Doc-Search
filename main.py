"""
Architecture overview — RAG document search API
================================================

Layers
------
- main.py: Application entrypoint. Builds the FastAPI app, mounts middleware and
  routers, and exposes ASGI to uvicorn.
- app/api: HTTP surface. Routes validate input/output via Pydantic models and
  delegate to services. Dependencies wire settings and service instances per request
  (or cached singletons where safe).
- app/models: Shared request/response schemas (DTOs). Keeps the API contract stable
  and separate from database row shapes.
- app/services: Domain logic. Ingestion (chunking, embedding), vector retrieval
  against PostgreSQL + pgvector, and optional answer generation (OpenAI + LangChain).
- app/core: Configuration and cross-cutting concerns (env loading, DB URL, API keys).

Data flow — ingestion (indexing)
---------------------------------
1. Raw documents enter through an ingest path (upload, crawl, or batch job — to be
   implemented in services).
2. Text is split into chunks; each chunk is embedded (e.g. OpenAI embeddings API).
3. Embeddings and metadata are stored in Postgres using the pgvector extension
   (typically via psycopg2 or LangChain’s vector store integrations).

Data flow — query (search / RAG)
----------------------------------
1. Client sends a natural language query to a route in app/api (e.g. POST /search).
2. The route parses the body into a Pydantic model (app/models) and calls RAGService.
3. RAGService embeds the query, runs a similarity search in pgvector (top-k chunks),
   and optionally builds a prompt with retrieved context.
4. An LLM (OpenAI, orchestrated with LangChain where helpful) returns an answer or
   the API returns ranked chunks for the client to consume.

This scaffold wires routing and dependency injection only; replace placeholders in
app/services with real DB schema, embedding model IDs, and LangChain chains.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.documents import router as documents_router
from app.api.routes import router as api_router
from app.core.config import get_settings

# Directory for the demo UI (HTML/CSS/JS). Kept next to main.py for a single tree.
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eager-load settings so misconfiguration fails fast at startup.
    _ = get_settings()
    yield


app = FastAPI(
    title="RAG Document Search API",
    lifespan=lifespan,
)
app.include_router(api_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1/documents")


@app.get("/")
async def serve_index() -> FileResponse:
    """Serve the single-page demo UI."""
    return FileResponse(STATIC_DIR / "index.html")


# Static file serving: Starlette's StaticFiles maps URL paths under ``/static`` to
# files on disk. FastAPI/uvicorn read and stream those bytes with cache-friendly
# headers. That is enough for **local testing** and small internal tools: zero
# build step, same origin as the API so the browser can call ``/api/v1/...``
# without CORS. In **production** you would usually put HTML/JS/CSS on a CDN or
# behind nginx, use hashed asset names and long cache lifetimes, TLS, gzip/br,
# and often a separate frontend host—serving arbitrary paths from the API
# process is simpler but not how public sites are typically run.
app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
