"""Phase 5, experiment 3: cross-encoder reranking. Tested independently
against baseline (not stacked with experiments 1 or 2) — same KB granularity
and same baseline kb_docs collection as retrieval_eval.py; only the ranking
of an already-retrieved candidate set changes.

Retrieve the top-10 candidates by vector similarity (retrieve.retrieve()),
then rerank them with cross-encoder/ms-marco-MiniLM-L-6-v2
(sentence-transformers), which scores each (query, candidate_answer) pair
jointly through a single transformer rather than comparing two independently
computed embeddings. That joint attention is why cross-encoders generally
outrank bi-encoder cosine similarity within a fixed candidate set — the model
can directly attend query tokens against candidate tokens — at the cost of
being too slow to run over the full KB (hence: rerank a bi-encoder's top-10,
don't replace the first-stage retrieval with it). The top-5 by cross-encoder
score become the final result, matching every other experiment's top_k.

Model choice: ms-marco-MiniLM-L-6-v2 is a small (~23M parameter), CPU-fast
cross-encoder trained on MS MARCO passage ranking — a reasonable off-the-shelf
choice for short-passage relevance ranking without needing a GPU. Verified
empirically before running the full eval: ~10s one-time model load, under 1s
to score 10 pairs, so 89 queries stay well under a minute end to end.

Run: python -m src.eval.experiment_reranker
"""
import json
from pathlib import Path

from sentence_transformers import CrossEncoder

from src.eval.retrieval_eval import TOP_K, compute_metrics, find_paraphrase_answer_ids, print_markdown_table, run_experiment
from src.retrieve import get_client, retrieve

EVAL_SET_PATH = Path("data/eval/eval_set.json")
RESULTS_PATH = Path("data/eval/retrieval_results_reranker.json")
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_DEPTH = 10


def make_retrieve_ids_fn(model: CrossEncoder, top_k: int = TOP_K, candidate_depth: int = CANDIDATE_DEPTH):
    def retrieve_ids(question: str) -> list[int]:
        candidates = retrieve(question, top_k=candidate_depth)
        pairs = [(question, c.answer) for c in candidates]
        scores = model.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda cs: -cs[1])
        return [c.answer_id for c, _ in reranked[:top_k]]

    return retrieve_ids


def main() -> None:
    model = CrossEncoder(RERANKER_MODEL)

    eval_rows = json.loads(EVAL_SET_PATH.read_text())
    paraphrase_answer_ids = find_paraphrase_answer_ids(eval_rows)

    records = run_experiment(eval_rows, retrieve_ids_fn=make_retrieve_ids_fn(model))
    get_client().close()

    aggregate = compute_metrics(records)
    paraphrase_records = [r for r in records if r["original_answer_id"] in paraphrase_answer_ids]
    paraphrase = compute_metrics(paraphrase_records)

    results = {
        "config": {"name": "cross_encoder_reranker", "model": RERANKER_MODEL, "candidate_depth": CANDIDATE_DEPTH},
        "aggregate": aggregate,
        "paraphrase_subset": {
            "note": "Illustrative only, not statistically meaningful: 3 answer_ids each shared by 2 differently-worded questions (n=6 queries total).",
            "answer_ids": sorted(paraphrase_answer_ids),
            "metrics": paraphrase,
        },
        "per_query": records,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print_markdown_table(aggregate, paraphrase, label="Experiment 3: cross-encoder reranker (top-10 -> top-5)")
    print(f"\nfull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
