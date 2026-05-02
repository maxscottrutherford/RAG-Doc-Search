"""
LLM-as-judge evaluation for RAG outputs
=======================================

Metrics (what each score means)
--------------------------------
**Relevance (1–5)** — Do the *retrieved chunks* actually address the user’s
question? High scores mean the passages are on-topic; low scores mean vector
search returned tangential or wrong material even before generation.

**Completeness (1–5)** — Given what was retrieved, could a good answer *in
principle* be formed? This captures whether the context jointly covers the
question (missing key facts → lower score), distinct from whether the model
wrote a good answer.

**Groundedness (1–5)** — Is the *generated answer* faithful to the retrieved
chunks? High scores mean claims appear supported by the passages; low scores
flag hallucination, contradictions, or facts imported from outside the context.

Why an evaluation layer stands out
----------------------------------
Most RAG demos stop at “it answered!” and never measure retrieval or honesty.
Without scores, teams cannot compare chunk sizes, embeddings, or prompts, and
users cannot trust outputs in production. A lightweight judge loop turns this
project into something you can **iterate on with numbers** and **audit** after
the fact—closer to a real product than a one-off demo.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Union

import psycopg2
from openai import OpenAI
from psycopg2.extras import Json
from pydantic import BaseModel, Field, field_validator

from app.core.config import Settings, get_settings
from app.models.schemas import SearchResultItem
from app.services.search import SimilarChunk

logger = logging.getLogger(__name__)

EVAL_MODEL = "gpt-4o"

ChunkInput = Union[SimilarChunk, SearchResultItem, Mapping[str, Any]]


class EvaluationScores(BaseModel):
    """Structured scores returned by the evaluator model and persisted to Postgres."""

    relevance: int = Field(ge=1, le=5, description="Retrieval quality vs the query")
    completeness: int = Field(ge=1, le=5, description="Whether context suffices to answer")
    groundedness: int = Field(ge=1, le=5, description="Answer fidelity to the chunks")
    notes: str | None = Field(default=None, description="Short rationale from the judge")

    @field_validator("relevance", "completeness", "groundedness", mode="before")
    @classmethod
    def _coerce_int(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("boolean not allowed")
        if isinstance(v, (int, float)):
            return int(round(float(v)))
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        raise ValueError(f"expected numeric score, got {type(v)}")


@dataclass(frozen=True)
class EvaluationResult:
    """Scores plus DB row id for the stored evaluation run."""

    scores: EvaluationScores
    evaluation_id: int
    raw_model_json: dict[str, Any]


def _serialize_chunks(chunks: Sequence[ChunkInput]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ch in chunks:
        if isinstance(ch, SimilarChunk):
            out.append(
                {
                    "id": ch.id,
                    "filename": ch.filename,
                    "chunk_index": ch.chunk_index,
                    "content": ch.content,
                    "similarity": ch.similarity,
                }
            )
        elif isinstance(ch, SearchResultItem):
            out.append(
                {
                    "chunk_id": ch.chunk_id,
                    "filename": ch.document_id,
                    "content": ch.text,
                    "similarity": ch.score,
                }
            )
        else:
            m = dict(ch)
            out.append(
                {
                    "chunk_id": m.get("chunk_id"),
                    "filename": m.get("filename") or m.get("document_id"),
                    "content": m.get("content") or m.get("text"),
                    "similarity": m.get("similarity") or m.get("score"),
                }
            )
    return out


def _build_evaluator_prompt(
    query: str,
    serialized_chunks: list[dict[str, Any]],
    answer: str,
) -> str:
    chunks_block = json.dumps(serialized_chunks, ensure_ascii=False, indent=2)
    return f"""You evaluate retrieval-augmented generation (RAG) outputs.

Score each dimension from 1 (worst) to 5 (best). Use only the information below.

**Relevance**: Are the retrieved chunks pertinent to answering the user's query?
**Completeness**: Does the combined context contain enough information that a correct answer could be formed (even if the model failed)?
**Groundedness**: Is the generated answer supported by the chunks, without invented facts that are not in the context?

User query:
{query}

Retrieved chunks (JSON):
{chunks_block}

Generated answer:
{answer}

Respond with a single JSON object and no other text, using this exact schema:
{{
  "relevance": <int 1-5>,
  "completeness": <int 1-5>,
  "groundedness": <int 1-5>,
  "notes": "<one or two sentences explaining the scores>"
}}
"""


def _call_evaluator(client: OpenAI, user_prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=EVAL_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a precise RAG evaluation assistant. Output only valid JSON.",
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)


def _persist_evaluation(
    conn,
    *,
    query: str,
    answer: str,
    chunks: list[dict[str, Any]],
    scores: EvaluationScores,
    model_response: dict[str, Any],
) -> int:
    sql = """
        INSERT INTO evaluations (
            query,
            answer,
            chunks,
            relevance,
            completeness,
            groundedness,
            model_response
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                query,
                answer,
                Json(chunks),
                scores.relevance,
                scores.completeness,
                scores.groundedness,
                Json(model_response),
            ),
        )
        row = cur.fetchone()
        if not row:
            raise RuntimeError("INSERT INTO evaluations did not return id")
        return int(row[0])


def evaluate_rag_output(
    query: str,
    chunks: Sequence[ChunkInput],
    answer: str,
    *,
    settings: Settings | None = None,
) -> EvaluationResult:
    """
    Score retrieval and generation with gpt-4o, then persist rows to ``evaluations``.

    Returns a structured score object (Pydantic) and the new database row id.
    """
    cfg = settings or get_settings()
    q = query.strip()
    serialized = _serialize_chunks(chunks)

    client = OpenAI(api_key=cfg.openai_api_key)
    user_prompt = _build_evaluator_prompt(q, serialized, answer)
    raw = _call_evaluator(client, user_prompt)

    scores = EvaluationScores.model_validate(raw)
    audit_payload = dict(raw)
    audit_payload["relevance"] = scores.relevance
    audit_payload["completeness"] = scores.completeness
    audit_payload["groundedness"] = scores.groundedness
    audit_payload["notes"] = scores.notes

    conn = psycopg2.connect(cfg.database_url)
    try:
        eval_id = _persist_evaluation(
            conn,
            query=q,
            answer=answer,
            chunks=serialized,
            scores=scores,
            model_response=audit_payload,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    logger.info(
        "RAG evaluation id=%s relevance=%s completeness=%s groundedness=%s",
        eval_id,
        scores.relevance,
        scores.completeness,
        scores.groundedness,
    )

    return EvaluationResult(
        scores=scores,
        evaluation_id=eval_id,
        raw_model_json=audit_payload,
    )
