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

**Retrieval experiment** (n = 89 eval queries, top_k = 5) — recall/MRR across candidate
configurations. No precision@k: this eval set has exactly one correct `answer_id` per
question, so precision@k collapses to a rescaled recall@k with no independent signal.

| Configuration                              | Recall@1 | Recall@3 | Recall@5 | MRR       |
|---------------------------------------------|----------|----------|----------|-----------|
| Baseline (raw vector, 1 answer = 1 chunk)    | 0.820    | 0.933    | 0.944    | 0.875     |
| + tuned chunk size (___ tok)                 | 0.__     | 0.__     | 0.__     | 0.__      |
| + hybrid (BM25 + vector)                     | 0.__     | 0.__     | 0.__     | 0.__      |
| + reranker                                   | **0.__** | **0.__** | **0.__** | **0.__**  |

*Paraphrase-pair subset* (3 `answer_id`s each shared by 2 differently-worded questions,
n = 6 — illustrative only, not statistically meaningful):

| Metric   | All queries (n=89) | Paraphrase-pair subset (n=6) |
|----------|---------------------|-------------------------------|
| Recall@1 | 0.820               | 0.500                        |
| Recall@3 | 0.933               | 0.667                         |
| Recall@5 | 0.944               | 0.667                         |
| MRR      | 0.875               | 0.583                         |
| Hit@1    | 0.820               | 0.500                         |

The paraphrase-pair subset lags the full set on every metric (e.g. Recall@1 0.50 vs.
0.82), a small early signal — n=6, not conclusive on its own — that reworded queries are
where this baseline is weakest, which is exactly what the hybrid and reranker passes in
this experiment are meant to target.

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
