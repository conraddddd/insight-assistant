# Insight — Evaluated RAG Assistant for Customer-Support Email Drafting

An AI system that reads an incoming customer-support email, retrieves the relevant
institutional knowledge, and drafts a grounded, source-cited reply. Runs **fully
locally** (no customer data leaves the machine) on Ollama + Qdrant.

> Built as a production-minded RAG system with a real evaluation harness — the focus
> is not just "it generates answers" but *measuring* whether those answers are correct
> and improving retrieval quality with evidence.

---

## Results

**Answer quality** (LLM-as-judge via qwen2.5:7b — a different model family from the
llama3 generator, to avoid self-preference bias — n = 89 test emails, 1–5 scale):

| Metric                                      | All queries (n=89) | Paraphrase-pair subset (n=6, illustrative) |
|----------------------------------------------|---------------------|----------------------------------------------|
| Faithfulness                                 | 4.820               | 4.833                                         |
| Relevance                                    | 4.191               | 4.167                                          |
| Completeness, all                            | 4.281               | 4.000                                          |
| Completeness, retrieval-hit subset (n=84)    | 4.298               | 4.000                                          |

Hallucinated citation rate: **0.0%** (0/143 citations — a deterministic set-membership
check against the retrieved context, not a judged score).

The faithfulness–relevance gap (4.82 vs. 4.19) is a generation-side signal: drafts are
well-grounded in retrieved context but not always fully targeted at what the customer
specifically asked — faithfulness alone would have missed this. The retrieval-hit
completeness gap is small (4.298 vs. 4.281 overall, n=84 of 89 with a retrieval hit) —
directional evidence that generation quality, not retrieval failure, is the larger lever
for improvement right now, though 5 misses is too small a sample to treat as conclusive.
The 0.0% hallucinated citation rate is direct evidence the grounded-prompt design in
`generate.py` holds under full evaluation, not just the two manual smoke tests it was
designed against. Phase 5 targets both open threads at once: the paraphrase-retrieval
weakness from the retrieval experiment above and the relevance gap here are plausibly the
same underlying issue, and a reranker — which reorders retrieved candidates by relevance
to the actual query rather than just surfacing more candidates — should plausibly improve
both.

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

*Takeaway:* Baseline is strong across the board; the remaining headroom is concentrated
in relevance and paraphrase-handling, both of which Phase 5's hybrid retrieval and
reranker experiments target.

---

## Architecture

```
Incoming email ─► embed query ─► Qdrant retrieval (top-k) ─► LLM draft (grounded + cited)
                                        ▲
                              knowledge base (atomic answers, embedded)
```

- **Embeddings:** mxbai-embed-large (via Ollama)
- **Vector store:** Qdrant
- **Generation:** llama3 (via Ollama), prompted to answer only from retrieved context and cite sources
- **Serving:** FastAPI `/draft` endpoint
- **Evaluation:** labeled test set + retrieval metrics + LLM-as-judge

## Setup

```bash
ollama pull llama3                   # generation model
ollama pull mxbai-embed-large        # embedding model
ollama pull qwen2.5:7b               # LLM-as-judge model (Phase 4b eval)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python -m src.ingest                 # embeds the index; Qdrant runs embedded/local
                                      # (no server) and creates qdrant_db/ on first run
uvicorn src.api:app --reload         # serve
```

## Evaluation

```bash
python -m src.eval.retrieval_eval    # recall@k, MRR, hit@1
python -m src.eval.answer_eval       # faithfulness, relevance, completeness (LLM-as-judge) + hallucinated citation rate
```

## Project structure

```
data/knowledge_base/   source KB documents
data/eval/             test emails + reference answers / relevant-doc labels
src/ingest.py          load, dedupe, embed, index
src/retrieve.py        retrieval
src/generate.py        grounded draft generation
src/api.py             FastAPI app
src/eval/retrieval_eval.py   recall@k, MRR, hit@1
src/eval/answer_eval.py      LLM-as-judge (faithfulness, relevance, completeness) + hallucinated citation rate
```
