"""
Vector search over ingested document chunks
===========================================

Cosine similarity (what it measures)
------------------------------------
Two vectors point in directions in high-dimensional space. **Cosine similarity**
measures how aligned those directions are (the cosine of the angle between them),
ignoring vector length when both are normalized. Text chunks with similar meaning
tend to land in similar directions after embedding, so a high cosine similarity
means the model judged the texts **semantically close** (same topic, paraphrase,
or related context)—not that they share the same words verbatim.

Why we embed the query the same way as the documents
----------------------------------------------------
Embeddings only live in a shared space if they come from the **same model** with
the **same settings** (here ``text-embedding-3-small`` at 1536 dimensions). If the
query were embedded with a different model or dimensionality, its vector would not
be comparable to the stored ``documents.embedding`` rows, and distance operators
would be meaningless. Matching ingestion and search keeps **apples-to-apples**
geometry for retrieval.

What the “similarity score” means in practice
---------------------------------------------
pgvector’s ``<=>`` operator returns **cosine distance** (smaller = closer). We
convert that to a **similarity** score as ``1 - (embedding <=> query_vector)``.
For typical **L2-normalized** embeddings (including OpenAI’s), cosine distance is
``1 - cosine_similarity``, so this score rises toward **1** for very aligned
chunks and falls toward **0** (or below in edge cases) for unrelated text. Use it
to **rank** chunks, not as a calibrated probability: treat it as “how strongly
the model associates this passage with the question,” subject to model and data limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg2
from openai import OpenAI
from pgvector.psycopg2 import register_vector

from app.core.config import Settings, get_settings
from app.services.ingestion import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL

DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class SimilarChunk:
    """One retrieved chunk with a cosine-based similarity score."""

    id: int
    filename: str
    content: str
    chunk_index: int
    similarity: float


def _embed_query(query: str, client: OpenAI) -> list[float]:
    """Return the embedding vector for a single search string."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return response.data[0].embedding


def search_similar_chunks(
    query: str,
    *,
    settings: Settings | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[SimilarChunk]:
    """
    Embed ``query`` and return the ``top_k`` chunks closest in cosine space.

    Uses pgvector ``<=>`` (cosine distance) in ``ORDER BY``; rows are ranked by
    increasing distance. ``similarity`` on each result is ``1 - distance`` so
    larger values indicate stronger semantic match for normalized embeddings.
    """
    stripped = query.strip()
    if not stripped:
        return []

    cfg = settings or get_settings()
    client = OpenAI(api_key=cfg.openai_api_key)
    query_vector = _embed_query(stripped, client)

    sql = """
        SELECT
            id,
            filename,
            content,
            chunk_index,
            1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    conn = psycopg2.connect(cfg.database_url)
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(sql, (query_vector, query_vector, top_k))
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        SimilarChunk(
            id=row[0],
            filename=row[1],
            content=row[2],
            chunk_index=row[3],
            similarity=float(row[4]),
        )
        for row in rows
    ]
