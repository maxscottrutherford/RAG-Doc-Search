"""RAG retrieval and generation orchestration."""

from openai import OpenAI

from app.core.config import Settings
from app.models.schemas import SearchRequest, SearchResponse, SearchResultItem
from app.services.search import SimilarChunk, search_similar_chunks

CHAT_MODEL = "gpt-4o"
RAG_TOP_K = 5


def build_rag_prompt(chunks: list[SimilarChunk], question: str) -> str:
    """
    Assemble a user message that lists retrieved passages and the question.

    Grounding in RAG means tying the model's answer to concrete retrieved text so
    generation is anchored in evidence from your index rather than free-form
    world knowledge alone. That reduces confident hallucinations when the corpus
    actually contains (or omits) the answer. Returning the same source chunks
    alongside the model output lets callers verify claims, spot bad retrieval
    (irrelevant passages ranked high), and debug prompts or data issues without
    trusting the summary blindly.
    """
    if not chunks:
        return (
            "No relevant passages were retrieved from the knowledge base.\n\n"
            f"Question: {question}\n\n"
            "Answer based on the fact that no context was provided."
        )

    lines: list[str] = [
        "Use ONLY the following passages from the knowledge base to answer the question. "
        "If the passages do not contain enough information, say so clearly.",
        "",
        "--- Context ---",
    ]
    for i, ch in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source file: {ch.filename}, chunk_index={ch.chunk_index}, similarity={ch.similarity:.4f})")
        lines.append(ch.content.strip())
        lines.append("")
    lines.append("--- End context ---")
    lines.append("")
    lines.append(f"Question: {question}")
    return "\n".join(lines)


class RAGService:
    """Coordinates embedding search (pgvector) and LLM answering."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def search(self, request: SearchRequest) -> SearchResponse:
        query = request.query.strip()
        chunks = search_similar_chunks(
            query,
            settings=self._settings,
            top_k=RAG_TOP_K,
        )

        source_chunks = [
            SearchResultItem(
                chunk_id=str(ch.id),
                document_id=ch.filename,
                text=ch.content,
                score=ch.similarity,
            )
            for ch in chunks
        ]

        if not chunks:
            return SearchResponse(
                query=query,
                answer=(
                    "No relevant passages were found in the knowledge base for this query. "
                    "Try uploading or ingesting documents, or rephrasing the question."
                ),
                source_chunks=[],
            )

        prompt = build_rag_prompt(chunks, query)
        client = OpenAI(api_key=self._settings.openai_api_key)

        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful assistant for document-grounded Q&A. "
                        "Cite which passage numbers ([1], [2], …) support your answer when possible. "
                        "Do not invent facts not supported by the context."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        answer = completion.choices[0].message.content or ""

        return SearchResponse(
            query=query,
            answer=answer.strip(),
            source_chunks=source_chunks,
        )


def get_rag_service(settings: Settings) -> RAGService:
    return RAGService(settings)
