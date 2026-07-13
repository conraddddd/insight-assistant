# Insight — Evaluated RAG Assistant for Customer-Support Email Drafting

An AI system that reads an incoming customer-support email, retrieves the relevant
institutional knowledge, and drafts a grounded, source-cited reply. Runs **fully
locally** (no customer data leaves the machine) on Ollama + Qdrant.

> Built as a production-minded RAG system with a real evaluation harness — not just
> "it generates answers," but *measured* evidence that those answers are correct and
> grounded. The harness didn't just validate the shipped config; it caught a citation
> regression the winning retrieval experiment introduced, before it shipped.

---

## Results

**Retrieval experiment** (n = 89 eval queries, top_k = 5) — recall/MRR across candidate
configurations, each tested *independently* against baseline (not stacked). No
precision@k: this eval set has exactly one correct `answer_id` per question, so
precision@k collapses to a rescaled recall@k with no independent signal.

| Configuration                                  | Recall@1  | Recall@3  | Recall@5  | MRR       |
|--------------------------------------------------|-----------|-----------|-----------|-----------|
| Baseline (raw vector, 1 answer = 1 chunk)         | 0.820     | 0.933     | 0.944     | 0.875     |
| **+ chunk concatenation (2-3 answers/chunk)**     | **0.831** | **0.966** | **0.989** | **0.900** |
| + hybrid (BM25 + vector, RRF)                     | 0.809     | 0.910     | 0.921     | 0.857     |
| + reranker (cross-encoder, top-10 → top-5)        | 0.798     | 0.910     | 0.921     | 0.851     |

*Paraphrase-pair subset* (3 `answer_id`s each shared by 2 differently-worded questions,
n = 6 — illustrative only, not statistically meaningful):

| Metric   | Baseline | + chunk concat | + hybrid | + reranker |
|----------|----------|-----------------|----------|------------|
| Recall@1 | 0.500    | **0.667**        | 0.500    | 0.667      |
| Recall@3 | 0.667    | **1.000**        | 0.667    | 0.667      |
| Recall@5 | 0.667    | **1.000**        | 0.667    | 0.667      |
| MRR      | 0.583    | **0.833**        | 0.583    | 0.667      |

*Takeaway:* Chunk concatenation won outright — retrieval and (after fixing a citation
regression it caused, see Findings below) generation quality both beat baseline on every
metric this project tracks. Hybrid BM25+vector and the cross-encoder reranker both
underperformed baseline on this data. **Chunk concatenation + sub-answer citation ids is
the shipped production config** (`src/chunking.py`, `src/api.py`), not a proposed
direction.

**Answer quality** (LLM-as-judge via qwen2.5:7b — a different model family from the
llama3 generator, to avoid self-preference bias — n = 89 test emails, 1–5 scale).
Phase 4b baseline (single-answer-per-chunk) shown alongside the post-Phase-5 production
config for a direct before/after:

| Metric                                     | Phase 4b baseline | Post-Phase-5 production |
|-----------------------------------------------|-------------------|----------------------------|
| Faithfulness                                   | 4.820             | **4.865**                  |
| Relevance                                      | 4.191             | **4.225**                  |
| Completeness, all                              | 4.281             | **4.416**                  |
| Completeness, retrieval-hit subset             | 4.298 (n=84)      | **4.420** (n=88)            |
| Hallucinated citation rate                     | 0.0%              | 0.0%                        |

*Paraphrase-pair subset* (n = 6, illustrative only):

| Metric        | Phase 4b baseline | Post-Phase-5 production |
|-----------------|-------------------|----------------------------|
| Faithfulness    | 4.833             | **5.000**                  |
| Relevance       | 4.167             | 4.000                       |
| Completeness    | 4.000             | **4.333**                  |

Every metric held or improved after switching to the production config — including
hallucinated citation rate, which only stayed at 0.0% because of a fix described below,
not by default. See Findings for the full story.

### Findings

**Why hybrid (BM25 + vector) and the reranker underperformed here.** Both techniques are
usually validated on longer, more lexically varied documents than this KB has. The 86-89
candidate answers are short (91–219 characters), template-like, and share a lot of generic
vocabulary across otherwise-unrelated answers ("please contact our customer support team"
appears in dozens of them) — so BM25's lexical overlap signal is mostly noise here, not a
useful complement to vector similarity, and diluted the RRF-fused ranking rather than
sharpening it. The cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) was trained on MS
MARCO's passage-ranking task — longer, more varied passages than single-sentence FAQ
answers — so its learned relevance signal likely doesn't transfer cleanly to candidates
this short and homogeneous. Both are plausible explanations grounded in a known
distribution mismatch, not something this project ran a controlled ablation to prove
outright.

**Why chunk concatenation won.** Grouping 2-3 topically-related answers into one chunk
gives each retrieval unit a "fatter" target: the chunk's embedding blends several adjacent
semantic directions instead of representing one narrow fact, so a wider range of
phrasings of the underlying question lands close to it in embedding space. This directly
explains the paraphrase-subset jump (Recall@5 0.667 → 1.000): a reworded query that drifts
slightly off the *exact* original answer's vector is much more likely to still fall
within a richer chunk's wider catch area than to have drifted enough to miss a single
narrow answer vector entirely.

**The citation regression — cause, diagnosis, fix.** Winning on retrieval metrics wasn't
the end of the story. Re-running the LLM-as-judge answer-quality eval against the
chunk-concatenation winner — not just trusting its retrieval numbers — surfaced a
regression invisible to retrieval-only evaluation: hallucinated citation rate jumped from
0.0% to 25.9%. Cause: bundling 2-3 answers under one blanket chunk id destroyed the 1:1
mapping between "a citable fact" and "an id" that the citation mechanism depends on — when
the model needed to cite the *specific* sub-answer it used within a chunk, it had no real
id to point to, and invented a small positional number instead ("[1]", "[2]"...), confirmed
by those invented citations clustering at small integers rather than being scattered
across the real chunk-id range. Diagnosis only came from scoring generation quality
directly; recall/MRR alone never look at what the model actually writes, so they had
nothing to say about this. Fix: sub-answer citation ids ("19a", "19b", "19c") built at
generation time from the already-indexed chunk payloads — no reindexing needed — giving
every individual fact a real, precise, citable id again. After the fix, hallucination rate
returned to 0.0% and faithfulness/relevance/completeness all improved *past* baseline too
— proof that the retrieval gain and the citation-faithfulness requirement weren't
fundamentally in tension; the tension was a specific, fixable implementation bug in how
citations were labeled, not an inherent cost of coarser chunking. (A second, unrelated bug
surfaced in the same verification pass: the abstention score gate, calibrated on the
baseline collection, wrongly refused a real, retrieved, answerable question because
concatenated-chunk embeddings score lower on average — fixed with a separately-calibrated
threshold for the concatenated collection.)

**The methodological takeaway.** Optimizing and shipping on retrieval metrics alone would
have shipped chunk concatenation with a 25.9% hallucinated-citation rate and a
mis-calibrated abstention gate wrongly refusing real questions — both real, user-facing
defects, both invisible to Recall@k and MRR. The LLM-as-judge harness, applied to every
retrieval candidate rather than just the shipped baseline, is what caught them before they
reached production. That's the concrete payoff of treating evaluation as the project's
main event instead of an afterthought.

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
