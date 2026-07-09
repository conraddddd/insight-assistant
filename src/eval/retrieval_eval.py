"""Phase 4: retrieval quality evaluation over data/eval/eval_set.json.

For each of the 89 eval questions, retrieve.retrieve() returns the top-5
nearest KB answers by raw cosine similarity — retrieve() has no score gate or
generation step attached to it (that gate lives entirely in generate.py), so
this measures retrieval alone, exactly as retrieved in production.

Let N be the number of eval queries, retrieved_i the top-5 answer_ids for
query i, gold_i the ground-truth answer_id, and rank_i the 1-indexed position
of gold_i within retrieved_i (undefined if gold_i is not in retrieved_i at
all — a miss).

    Recall@k = (1/N) * sum_i  1[ gold_i in retrieved_i[:k] ]     for k in {1,3,5}

    MRR      = (1/N) * sum_i  RR_i,   RR_i = 1/rank_i if gold_i in retrieved_i
                                             = 0        otherwise
             = (1/N) * ( RR_1 + RR_2 + ... + RR_N )

    Hit@1    = (1/N) * sum_i  1[ retrieved_i[0] == gold_i ]

A miss (gold_i not in the top-5 at all) contributes 0 to every metric above —
it is never dropped from N or excluded from the average, since doing so would
silently hide exactly the failure cases these metrics exist to surface.

Why these three, and not precision@k: this eval set has exactly one correct
answer_id per question, so precision@k collapses to 1/k (hit) or 0/k (miss) —
it's a rescaled, noisier version of the same signal recall@k already
captures, with no independent information. Recall@k, MRR, and Hit@1 each add
something recall@k alone doesn't: Recall@k shows whether the right answer
shows up at all within a candidate window (useful if a reranker will sit
downstream); MRR captures *where* within that window, rewarding rank 1 over
rank 5 rather than treating them as equally good; Hit@1 is the strictest and
most user-facing cut, since with top_k=5 for generation the model still sees
ranks 2-5, but a real single-shot "did we get it right immediately" read is
still worth reporting on its own.

Note: with exactly one gold answer_id per question, Hit@1 and Recall@1 are
the identical formula (1[retrieved_i[0] == gold_i] is exactly the k=1 case of
1[gold_i in retrieved_i[:k]]). Both are still reported because they read as
two different questions to a hiring manager ("is rank-1 accuracy on its own
worth calling out" vs "what's the k=1 point of the recall curve") even though
the number is mathematically forced to match.

A secondary breakdown covers the 3 paraphrase pairs discovered during
ingestion (data/ingest.py's dedupe_and_split docstring) - 3 answer_ids each
shared by two differently-worded questions, i.e. 6 of the 89 queries. This is
reported separately and marked explicitly as illustrative, not statistical:
n=6 is nowhere near large enough to draw a real conclusion, it's just a
direct look at whether retrieval is robust to paraphrasing on the few cases
where the dataset happens to test that.

Run: python -m src.eval.retrieval_eval
"""
import json
from collections import defaultdict
from pathlib import Path

from src.retrieve import get_client, retrieve

EVAL_SET_PATH = Path("data/eval/eval_set.json")
RESULTS_PATH = Path("data/eval/retrieval_results.json")
TOP_K = 5
RECALL_KS = (1, 3, 5)


def evaluate_query(question: str, gold_answer_id: int, k: int = TOP_K) -> dict:
    chunks = retrieve(question, top_k=k)
    retrieved_ids = [c.answer_id for c in chunks]
    rank = retrieved_ids.index(gold_answer_id) + 1 if gold_answer_id in retrieved_ids else None
    return {
        "question": question,
        "gold_answer_id": gold_answer_id,
        "retrieved_ids": retrieved_ids,
        "rank": rank,
        "reciprocal_rank": 1 / rank if rank is not None else 0.0,
    }


def compute_metrics(records: list[dict], ks: tuple[int, ...] = RECALL_KS) -> dict:
    n = len(records)
    metrics = {f"recall@{k}": sum(1 for r in records if r["rank"] is not None and r["rank"] <= k) / n for k in ks}
    metrics["mrr"] = sum(r["reciprocal_rank"] for r in records) / n
    metrics["hit@1"] = sum(1 for r in records if r["rank"] == 1) / n
    metrics["n"] = n
    return metrics


def find_paraphrase_answer_ids(eval_rows: list[dict]) -> set[int]:
    """answer_ids shared by more than one distinct question in the eval set."""
    questions_by_answer = defaultdict(set)
    for row in eval_rows:
        questions_by_answer[row["answer_id"]].add(row["question"])
    return {answer_id for answer_id, qs in questions_by_answer.items() if len(qs) > 1}


def print_markdown_table(aggregate: dict, paraphrase: dict) -> None:
    rows = [
        ("Recall@1", "recall@1"),
        ("Recall@3", "recall@3"),
        ("Recall@5", "recall@5"),
        ("MRR", "mrr"),
        ("Hit@1", "hit@1"),
    ]
    print(f"| Metric | All queries (n={aggregate['n']}) | Paraphrase-pair subset (n={paraphrase['n']}, illustrative) |")
    print("|---|---|---|")
    for label, key in rows:
        print(f"| {label} | {aggregate[key]:.3f} | {paraphrase[key]:.3f} |")
    print()
    print("Hit@1 ≡ Recall@1 above: with exactly one gold answer per query, the two formulas coincide.")
    print("Paraphrase-pair subset is 3 answer_ids × 2 differently-worded questions each (n=6) — illustrative only, not statistically meaningful.")


def main() -> None:
    eval_rows = json.loads(EVAL_SET_PATH.read_text())
    paraphrase_answer_ids = find_paraphrase_answer_ids(eval_rows)

    records = [evaluate_query(row["question"], row["answer_id"]) for row in eval_rows]
    get_client().close()

    aggregate = compute_metrics(records)
    paraphrase_records = [r for r in records if r["gold_answer_id"] in paraphrase_answer_ids]
    paraphrase = compute_metrics(paraphrase_records)

    results = {
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

    print_markdown_table(aggregate, paraphrase)
    print(f"\nfull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
