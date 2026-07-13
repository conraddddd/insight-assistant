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
Incoming email ─► embed query ─► Qdrant retrieval (top-5 chunks) ─► LLM draft
                                        ▲                    (grounded + sub-answer cited, e.g. "19a")
                    knowledge base (2-3 topically-grouped answers/chunk, embedded)
```

- **Embeddings:** mxbai-embed-large (via Ollama)
- **Vector store:** Qdrant, embedded/local. Two collections: `kb_docs_concat` (production —
  2-3 topically-related answers per chunk, grouped via greedy nearest-neighbor chaining
  over embeddings, `src/chunking.py`) and `kb_docs` (baseline — 1 answer per chunk,
  `src/ingest.py`, kept only as the Phase 4/5 eval reference point)
- **Generation:** llama3 (via Ollama), prompted to answer only from retrieved context and
  cite sources by sub-answer id (`"19a"`, `"19b"`, ...) — one id per original answer within
  a chunk, not one blanket id per chunk (see README Findings above for why that distinction
  matters)
- **Serving:** FastAPI `/draft` endpoint, serving from `kb_docs_concat`
- **Evaluation:** labeled test set + retrieval metrics + LLM-as-judge, run both as a
  standalone baseline and against every Phase 5 retrieval candidate

## Setup

```bash
ollama pull llama3                   # generation model
ollama pull mxbai-embed-large        # embedding model
ollama pull qwen2.5:7b               # LLM-as-judge model (Phase 4b/5 eval)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python -m src.ingest                 # builds kb_docs (baseline, eval reference point)
python -m src.chunking                # builds kb_docs_concat (production collection) — a
                                       # separate step, not folded into ingest: different
                                       # grouping algorithm, different collection, kept
                                       # independently reproducible/comparable
uvicorn src.api:app --reload         # serve — reads from kb_docs_concat
```

## Evaluation

```bash
# Baseline (single-answer-per-chunk collection)
python -m src.eval.retrieval_eval        # recall@k, MRR, hit@1
python -m src.eval.answer_eval           # faithfulness, relevance, completeness (LLM-as-judge) + hallucinated citation rate

# Phase 5 retrieval experiments (each independent vs. baseline, not stacked)
python -m src.eval.experiment_chunking   # chunk concatenation — builds kb_docs_concat, then scores it
python -m src.eval.experiment_hybrid     # hybrid BM25 + vector (RRF)
python -m src.eval.experiment_reranker   # cross-encoder reranker (top-10 -> top-5)

# Answer quality of the shipped production config (chunk concat + sub-answer citations)
python -m src.eval.answer_eval_chunking
```

## Project structure

```
data/knowledge_base/   source KB documents
data/eval/             test emails + reference answers / relevant-doc labels
src/ingest.py                    load, dedupe, embed, index — baseline (kb_docs), eval reference point
src/chunking.py                  chunk-concatenated production KB: grouping, indexing, sub-answer
                                  citation context — what src/api.py actually serves from
src/retrieve.py                  retrieval (embed_query/get_client shared by src/chunking.py too)
src/generate.py                  grounded draft generation — baseline + sub-answer citation prompts
src/api.py                       FastAPI app, serves from the production chunk-concat config
src/eval/retrieval_eval.py       recall@k, MRR, hit@1 (baseline) + scoring helpers every Phase 5
                                  experiment reuses
src/eval/answer_eval.py          LLM-as-judge (baseline) + hallucinated citation rate
src/eval/experiment_chunking.py  Phase 5: chunk concatenation retrieval eval
src/eval/experiment_hybrid.py    Phase 5: hybrid BM25 + vector (RRF) retrieval eval
src/eval/experiment_reranker.py  Phase 5: cross-encoder reranker retrieval eval
src/eval/answer_eval_chunking.py Phase 5: LLM-as-judge eval against the production config
```

## Extending to production

The portfolio version prioritizes measurability and reproducibility over deployment
concerns; a production deployment would address the following:

- **Ingestion trigger and incrementality.** Right now ingestion is a one-shot batch job,
  triggered by hand (`python -m src.ingest`, `python -m src.chunking`) and rebuilding each
  collection from scratch every time (`build_collection` deletes and recreates it).
  Production ingestion would instead be scheduled, event-driven (e.g. a webhook firing
  when a source KB article changes), or API-triggered, and — more importantly —
  incremental: detecting which documents actually changed since the last run and
  upserting only those, rather than re-embedding and reindexing the entire KB on every
  run regardless of what changed.

- **Customer-facing citation transformation.** The `[19a]`-style bracket markers in a
  draft are internal scaffolding — they exist for auditability and so the eval harness can
  check for hallucinated citations, not because a customer should see raw bracket ids in
  their inbox. A production system would strip them before sending, convert them to
  footnote-style links pointing at the source KB article, or keep them visible alongside a
  source panel; which of these is right depends on the domain — a regulated industry might
  want visible sourcing by default, a consumer product might want the reply to read as
  plain, unannotated prose.

- **Spoofed-citation policy.** Currently, `cited_ids`/`cited_sub_ids` silently filters any
  hallucinated citation out of the response at the API layer (`src/api.py`) — if the model
  cites something that wasn't actually retrieved, it just quietly disappears from what the
  caller sees, with no signal that generation produced a bad citation in the first place.
  Production alternatives include regenerating the draft automatically when a hallucinated
  citation is detected, failing closed (refusing to return a draft at all rather than
  risk silently editing one), or flagging the reply for human review before it ever
  reaches the customer — the right choice is domain-dependent, e.g. healthcare or finance
  would likely favor fail-closed or human review over silent filtering.

- **Monitoring hooks.** The evaluation harness (`src/eval/`) currently only runs at
  development time, invoked by hand against the fixed 89-query eval set. A production
  deployment would run the same evaluation code continuously against a sampled slice of
  real traffic, with alerting wired to score regressions — faithfulness dropping, or
  hallucination rate crossing a threshold — so a degradation is caught automatically
  instead of only being visible the next time someone happens to rerun the eval scripts by
  hand.
