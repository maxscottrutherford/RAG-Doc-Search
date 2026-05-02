"""Ingest raw text: token chunking, OpenAI embeddings, and Postgres storage."""

from __future__ import annotations

import logging
import psycopg2
import tiktoken
from openai import OpenAI
from pgvector.psycopg2 import register_vector

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50
EMBEDDING_BATCH_SIZE = 100


def _get_token_encoder() -> tiktoken.Encoding:
    """Return the tiktoken encoder used for chunk boundaries.

    What it does
    ------------
    Resolves a tiktoken encoding compatible with OpenAI embedding models so chunk
    sizes are counted in real tokens (not characters), matching how the model sees text.

    Why chunking with overlap matters (for the pipeline this feeds)
    ---------------------------------------------------------------
    Overlap gives neighboring chunks shared context at boundaries so ideas that span
    a split are still partially present in both chunks. That reduces cases where the
    only relevant sentence sits exactly on a hard boundary and would be weakened or
    lost for retrieval. Token-accurate windows keep each chunk under the model’s
    context limits and comparable in semantic size.

    What the embedding vector represents (in this system)
    ------------------------------------------------------
    Embeddings are not produced here; this step only tokenizes. Later, each chunk’s
    embedding will be a fixed-length list of floats (here length 1536) representing
    that chunk’s meaning in model space for similarity search in pgvector.
    """
    try:
        return tiktoken.encoding_for_model(EMBEDDING_MODEL)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str,
    *,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[str]:
    """Split raw text into overlapping token windows and decode back to strings.

    What it does
    ------------
    Encodes the document to token IDs, walks the sequence in steps of
    ``chunk_tokens - overlap_tokens``, and decodes each window of at most
    ``chunk_tokens`` tokens back to UTF-8 text. Returns one string per chunk in order.

    Why chunking with overlap is important
    ----------------------------------------
    Without overlap, a fact split across two adjacent chunks might not appear fully in
    either chunk, hurting retrieval. Reusing the last ~50 tokens at each step keeps
    boundary context duplicated so nearest-neighbor search can still surface the right
    region. Overlap trades a bit of storage for more robust recall at section edges.

    What the embedding vector represents
    ------------------------------------
    Each resulting string will later be embedded as a vector: a numeric summary of the
    chunk’s semantics learned by ``text-embedding-3-small``. Similar vectors indicate
    similar meaning, which is how pgvector ranks chunks for a user query.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    enc = _get_token_encoder()
    token_ids = enc.encode(cleaned)
    if not token_ids:
        return []

    step = max(chunk_tokens - overlap_tokens, 1)
    chunks: list[str] = []
    start = 0
    while start < len(token_ids):
        window = token_ids[start : start + chunk_tokens]
        if window:
            chunks.append(enc.decode(window))
        start += step

    return chunks


def embed_chunks(chunks: list[str], client: OpenAI) -> list[list[float]]:
    """Call OpenAI embeddings for each chunk, preserving order.

    What it does
    ------------
    Sends text chunks to ``text-embedding-3-small`` in batches (up to
    ``EMBEDDING_BATCH_SIZE`` strings per request) and concatenates the returned
    embedding lists in the same order as ``chunks``.

    Why chunking with overlap is important (for these inputs)
    ---------------------------------------------------------
    Embeddings are computed per chunk; overlap ensures the text passed into this
    function still carries boundary context from the prior window. That makes each
    vector a better local summary when an answer spans a nominal split point.

    What the embedding vector represents
    ------------------------------------
    Each vector is the model’s 1536-dimensional representation of one chunk’s meaning.
    Dot products / cosine similarity in this space approximate semantic similarity;
    pgvector stores these so queries can retrieve the closest chunks to a question’s
    embedding.
    """
    if not chunks:
        return []

    all_embeddings: list[list[float]] = []
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        # API returns data in the same order as input.
        ordered = sorted(response.data, key=lambda d: d.index)
        all_embeddings.extend(item.embedding for item in ordered)

    if len(all_embeddings) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(all_embeddings)}, expected {len(chunks)}"
        )
    return all_embeddings


def store_chunks(
    conn,
    filename: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> int:
    """Insert chunk rows into ``documents`` inside an existing transaction.

    What it does
    ------------
    Registers pgvector on the connection, then inserts ``(filename, content,
    embedding, chunk_index)`` for each chunk/embedding pair.

    Why chunking with overlap is important (for stored rows)
    ---------------------------------------------------------
    Overlapping chunks mean some text appears in multiple rows with different
    ``chunk_index`` values. That redundancy is intentional: retrieval can return
    whichever overlapping row aligns best with the query vector, reducing missed hits
    at chunk edges.

    What the embedding vector represents
    ------------------------------------
    The ``embedding`` column stores the same 1536-float semantic vector returned by
    OpenAI for the adjacent ``content`` text; pgvector indexes/query operators use it
    for nearest-neighbor search against future query embeddings.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    register_vector(conn)
    inserted = 0
    with conn.cursor() as cur:
        for idx, (content, embedding) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """
                INSERT INTO documents (filename, content, embedding, chunk_index)
                VALUES (%s, %s, %s, %s)
                """,
                (filename, content, embedding, idx),
            )
            inserted += 1
    return inserted


def ingest_text_document(
    text: str,
    filename: str,
    *,
    settings: Settings | None = None,
) -> int:
    """Chunk a document, embed each chunk, and persist rows to PostgreSQL.

    What it does
    ------------
    Loads settings (including ``OPENAI_API_KEY`` and ``DATABASE_URL`` from the
    environment via ``python-dotenv``), splits ``text`` into overlapping token chunks,
    obtains embeddings from OpenAI, opens a psycopg2 connection, and inserts all rows
    in one transaction.

    Why chunking with overlap is important
    --------------------------------------
    Long documents exceed embedding context limits and mix many topics in one vector.
    Smaller overlapped windows localize semantics and soften boundary effects so RAG
    retrieves the right passage more often than disjoint huge splits or zero-overlap
    slices would.

    What the embedding vector represents
    ------------------------------------
    For each chunk, the vector is a dense semantic code produced by
    ``text-embedding-3-small``: direction in 1536-D space encodes topic and wording
    similarity. The database uses these vectors to find chunks whose meaning is
    closest to a user’s question embedding.
    """
    cfg = settings or get_settings()
    chunks = chunk_text(text)
    if not chunks:
        logger.info("No chunks produced for empty or whitespace-only document: %s", filename)
        return 0

    client = OpenAI(api_key=cfg.openai_api_key)
    embeddings = embed_chunks(chunks, client)

    conn = psycopg2.connect(cfg.database_url)
    try:
        inserted = store_chunks(conn, filename, chunks, embeddings)
        conn.commit()
        return inserted
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
