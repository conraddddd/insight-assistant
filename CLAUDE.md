# Insight — Evaluated RAG Assistant (CV Portfolio Project)

## Purpose & context
This is a portfolio project for an AI engineer job application. The owner (Conrad) has a
CS + cybersecurity background (CEH certified, ex-bank IT officer) and is building this
from scratch to learn — **explain design decisions as you go; don't just generate code.**
He needs to be able to defend every line in an interview.

The differentiator vs. typical "RAG chatbot" portfolio projects is the **evaluation
harness and a documented retrieval experiment with before/after metrics**. Treat Phases
4–5 as the main event, not an afterthought.

## What it does
Reads an incoming customer-support email → retrieves relevant knowledge from a vector
store → drafts a grounded, source-cited reply. Runs fully locally (privacy angle ties
into owner's security background).

## Stack (decided — don't change without discussion)
- **Generation:** llama3 via Ollama (localhost:11434)
- **Embeddings:** mxbai-embed-large via Ollama
- **Vector store:** Qdrant in **embedded/local mode** — `QdrantClient(path="./qdrant_db")`.
  NO Docker, NO server. The `docker-compose.yml` in the repo is unused; ignore it.
  Note: `src/config.py` still has a `qdrant_url` field from before this decision —
  replace it with a local path setting when writing ingest.
- **Serving:** FastAPI + uvicorn
- **Python 3.14.5**, deps pinned in requirements.txt (already frozen; qdrant-client 1.18)

## Critical API note
qdrant-client 1.18 removed deprecated methods (`search`, `recommend`, `search_batch`).
Use `query_points` for querying. Older tutorials online use the removed API — don't copy them.

## Dataset
`data/faqs.json` = MakTek/Customer_support_faqs_dataset from Hugging Face
(200 Q/A pairs, JSON, fields: `question`, `answer`, Apache-2.0).
Split plan: **answers become the knowledge base; questions become test emails;
the Q→A pairing gives free ground-truth labels for retrieval eval.**
Hold out the question→answer mapping for evaluation — don't leak test questions into the index.

## Phase roadmap (currently starting Phase 2)
1. ~~Setup~~ ✅ (repo scaffold, deps installed, Ollama models pulled: llama3, mxbai-embed-large)
2. **Ingestion** (`src/ingest.py`) — load faqs.json, split KB vs test set, chunk, embed, index
3. **Core RAG** (`src/retrieve.py`, `src/generate.py`, `src/api.py`) — retrieval + grounded
   cited drafts, FastAPI `/draft` endpoint
4. **Evaluation harness** (`src/eval/`) — retrieval metrics (recall@k, precision@k) +
   LLM-as-judge (faithfulness, relevance, completeness)
5. **Retrieval experiment** — baseline vs chunk-size tuning vs hybrid BM25+vector
   (rank-bm25 is installed) vs reranker. Record results in README table.
6. Deploy polish + README results write-up

## README convention
README.md leads with a Results section containing placeholder tables — fill them with
real numbers as Phases 4–5 produce them. Never fabricate metrics.

## Conventions
- Config via `src/config.py` (pydantic-settings) + `.env`; class defaults are fallbacks,
  `.env` overrides. Add a `.gitignore` (with `.env`, `qdrant_db/`, `.venv/`) before first push.
- Keep modules small and explanations in commit messages — this repo will be read by hiring managers.
