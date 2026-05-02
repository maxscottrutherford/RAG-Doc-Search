"""Document upload and ingestion routes."""

from __future__ import annotations

import os
from pathlib import Path

import anyio
import fitz  # PyMuPDF
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import settings_dep
from app.core.config import Settings
from app.models.schemas import DocumentUploadResponse
from app.services.ingestion import ingest_text_document

router = APIRouter()

_ALLOWED_SUFFIXES = {".txt", ".pdf"}


def _safe_filename(name: str | None) -> str:
    base = os.path.basename(name) if name else "upload"
    return base or "upload"


def _extract_text_from_upload(filename: str, raw: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        # Plain text: decode bytes as UTF-8; replace invalid sequences so uploads
        # still ingest instead of failing the whole request on one bad byte.
        return raw.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        # PyMuPDF opens the PDF from memory and concatenates per-page text.
        doc = fitz.open(stream=raw, filetype="pdf")
        try:
            parts: list[str] = []
            for page in doc:
                parts.append(page.get_text())
        finally:
            doc.close()
        return "\n".join(parts)
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type {suffix!r}; allowed: .txt, .pdf",
    )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    # Multipart file upload: the client sends an HTTP body with Content-Type
    # multipart/form-data. That format splits the request into "parts" separated
    # by a boundary string so the server can receive named fields and one or more
    # files in a single request (unlike JSON, which is not ideal for raw bytes).
    # FastAPI/Starlette parses those parts and gives us an UploadFile handle.
    file: UploadFile = File(..., description="A .txt or .pdf document to index"),
    settings: Settings = Depends(settings_dep),
) -> DocumentUploadResponse:
    filename = _safe_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}; allowed: {', '.join(sorted(_ALLOWED_SUFFIXES))}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file upload")

    text = _extract_text_from_upload(filename, raw).strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail="No extractable text (file may be empty or PDF may have no text layer)",
        )

    # We chunk before embedding because a whole long document becomes one blurred
    # vector: retrieval cannot pinpoint a paragraph, and models have input limits.
    # Smaller chunks get embeddings that each represent a local meaning, so vector
    # search returns the right passage for a question instead of one generic summary.
    def _run_ingest() -> int:
        return ingest_text_document(text, filename, settings=settings)

    chunks_stored = await anyio.to_thread.run_sync(_run_ingest)

    return DocumentUploadResponse(filename=filename, chunks_stored=chunks_stored)
