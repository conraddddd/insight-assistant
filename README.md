# Insight — Evaluated RAG Assistant for Customer-Support Email Drafting

An AI system that reads an incoming customer-support email, retrieves the relevant
institutional knowledge, and drafts a grounded, source-cited reply. Runs **fully
locally** (no customer data leaves the machine) on Ollama + Qdrant.

> Built as a production-minded RAG system with a real evaluation harness — the focus
> is not just "it generates answers" but *measuring* whether those answers are correct
> and improving retrieval quality with evidence.

---

## Results

*(Fill these in after Phases 4–5. This section is what people read first.)*

**Answer quality** (LLM-as-judge, n = __ test emails, 1–5 scale):

| Metric        | Score |
|---------------|-------|
| Faithfulness  | _._   |
| Relevance     | _._   |
| Completeness  | _._   |

**Retrieval experiment** — effect of each change on recall@5:

| Configuration                    | Recall@5 | Precision@5 |
|----------------------------------|----------|-------------|
| Baseline (naive vector, 512-tok) | 0.__     | 0.__        |
| + tuned chunk size (___ tok)     | 0.__     | 0.__        |
| + hybrid (BM25 + vector)         | 0.__     | 0.__        |
| + reranker                       | **0.__** | **0.__**    |

*Takeaway:* _one sentence on what moved the needle and why._

---

## Architecture

```
Incoming email ─► embed query ─► Qdrant retrieval (top-k) ─► LLM draft (grounded + cited)
                                        ▲
                              knowledge base (chunked + embedded)
```

- **Embeddings:** mxbai-embed-large (via Ollama)
- **Vector store:** Qdrant
- **Generation:** llama3 (via Ollama), prompted to answer only from retrieved context and cite sources
- **Serving:** FastAPI `/draft` endpoint
- **Evaluation:** labeled test set + retrieval metrics + LLM-as-judge

## Setup

```bash
docker compose up -d                 # start Qdrant
ollama pull llama3                   # generation model
ollama pull mxbai-embed-large        # embedding model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python -m src.ingest                 # build the index
uvicorn src.api:app --reload         # serve
```

## Evaluation

```bash
python -m src.eval.retrieval_eval    # recall@k / precision@k
python -m src.eval.answer_eval       # LLM-as-judge on draft quality
```

## Project structure

```
data/knowledge_base/   source KB documents
data/eval/             test emails + reference answers / relevant-doc labels
src/ingest.py          chunk + embed + index
src/retrieve.py        retrieval
src/generate.py        grounded draft generation
src/api.py             FastAPI app
src/eval/              retrieval metrics + LLM-as-judge
```
