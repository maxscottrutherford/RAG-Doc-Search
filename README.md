# RAG Document Search API

A small but end-to-end **retrieval-augmented generation (RAG)** service: upload documents, index them into **PostgreSQL + pgvector**, ask questions in natural language, and get answers grounded in your corpus—with **source chunks**, **similarity scores**, and an optional **LLM-as-judge evaluation** persisted for later analysis. The stack is **FastAPI**, **OpenAI** (embeddings + chat), **vanilla JS** demo UI, and **Docker** for the database.

---

## What RAG is and why it matters

Large language models can produce fluent text, but they do not know your private documents, your latest internal policies, or facts that appeared after their training cutoff. **Retrieval-augmented generation** bridges that gap by treating the model as a *reasoner over evidence* rather than a standalone oracle. At query time, the system retrieves a small set of passages from a corpus you control, injects them into the prompt, and asks the model to answer *using that context*. The model’s job shifts from memorizing the world to synthesizing and citing what you actually have on file.

That design choice matters for real products. RAG reduces blatant hallucination on domain-specific questions when retrieval succeeds, and it gives you a lever you can measure: better chunking, better embeddings, or better reranking directly improve what the model sees. It also creates an audit trail—if you return the retrieved chunks alongside the answer, users and engineers can verify whether the response was **grounded** in the right material or whether search surfaced irrelevant noise.

RAG is not a silver bullet. Retrieval can miss the right passage, chunk boundaries can split important sentences, and models can still ignore or misread context. Treating RAG as a *pipeline* (ingest → embed → store → retrieve → generate → optionally evaluate) forces you to think about each stage. This project implements that pipeline in a form you can run locally, extend, and discuss in interviews: concrete API boundaries, SQL you can inspect, and a simple UI to demo the behavior.

---

## Architecture

```
INGESTION FLOW                          QUERY (RAG) FLOW
================                        =================

  .txt / .pdf                           User question (HTTP / UI)
       |                                         |
       v                                         v
  Extract text                          Embed query (OpenAI)
  (UTF-8 / PyMuPDF)                     text-embedding-3-small
       |                                         |
       v                                         v
  Chunk by tokens                       pgvector: cosine distance
  (~500 tok, 50 overlap)                (<=>) on documents.embedding
       |                                         |
       v                                         v
  Embed each chunk                      Top-k rows from documents
  (same model + 1536 dims)                     |
       |                                         v
       v                                  Build prompt + context
  INSERT INTO documents                   (grounded generation)
  (content, embedding, ...)                      |
       |                                         v
       v                                  Chat completion (gpt-4o)
  PostgreSQL 15                                |
  + pgvector extension                         v
                                         Return: answer +
                                         source_chunks (+ eval scores
                                         -> evaluations table)

  Static UI (GET /) ----fetch---->  POST /api/v1/documents/upload
                              \-->  POST /api/v1/search
```

At a high level, **ingestion** turns files into overlapping chunks, embeds them, and stores vectors in Postgres. **Query** embeds the question the same way, runs a vector similarity search, assembles a prompt, and calls the chat model. A separate **evaluation** step scores retrieval and groundedness and logs to an `evaluations` table.

---

## Tech stack

| Layer | Choice |
|--------|--------|
| API | FastAPI, Pydantic, Uvicorn |
| Database | PostgreSQL 15, pgvector, psycopg2 |
| Embeddings | OpenAI `text-embedding-3-small` (1536-d), tiktoken chunking |
| Chat | OpenAI `gpt-4o` |
| Ingestion | PyMuPDF (PDF), chunked insert pipeline |
| Evaluation | gpt-4o JSON judge → `evaluations` table |
| UI | Static HTML/CSS/JS served from FastAPI |

---

## Prerequisites

- **Docker Desktop** (or Docker Engine + Compose) for Postgres
- **Python 3.11+** recommended (3.13 works in development)
- An **OpenAI API key** with access to the models above

---

## Setup

### 1. Clone and install Python dependencies

```bash
git clone <your-repo-url>
cd RAG-Doc-Search
pip install -r requirements.txt
```

On Windows, if the `uvicorn` command is not on your PATH, use `python -m uvicorn` (see run instructions below).

### 2. Configure environment

Copy the example env file and set your key:

```bash
cp .env.example .env
```

Edit `.env`:

- **`DATABASE_URL`** — Must match Docker Compose. The default in `.env.example` uses host **`127.0.0.1`** and port **`5433`** so the container does not conflict with a local Postgres on `5432`.
- **`OPENAI_API_KEY`** — Your secret key (never commit `.env`).

### 3. Start PostgreSQL

From the project root:

```bash
docker compose up -d
```

First boot applies `init.sql` (pgvector extension, `documents`, `evaluations` tables). Data persists in the `postgres_data` volume until you run `docker compose down -v`.

If you created the database **before** `evaluations` existed, create the table once (see `init.sql`) or recreate the volume.

### 4. Run the API

```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- **Interactive API docs:** http://127.0.0.1:8000/docs  
- **Demo UI:** http://127.0.0.1:8000/  

---

## Example: curl

Replace paths and host as needed. These assume the API on port **8000**.

**Upload a text file** (multipart form, field name `file`):

```bash
curl -s -X POST "http://127.0.0.1:8000/api/v1/documents/upload" \
  -F "file=@./sample.txt"
```

Example success payload:

```json
{"filename":"sample.txt","chunks_stored":3}
```

**Search (RAG + optional evaluation):**

```bash
curl -s -X POST "http://127.0.0.1:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What does the document say about refunds?\"}"
```

The response includes **`answer`**, **`source_chunks`** (with **`score`** similarity), and usually **`evaluation`** (1–5 relevance, completeness, groundedness) unless evaluation fails or no chunks were retrieved.

---

## Project layout (abbreviated)

```
├── main.py                 # FastAPI app, static UI mount
├── docker-compose.yml      # Postgres + pgvector
├── init.sql                # Schema: documents, evaluations
├── requirements.txt
├── static/                 # index.html, styles.css, app.js
└── app/
    ├── api/                # routes, document upload
    ├── core/               # settings / .env
    ├── models/             # Pydantic schemas
    └── services/           # ingestion, search, RAG, evaluation
```

