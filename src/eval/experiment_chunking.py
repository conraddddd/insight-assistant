"""Phase 5, experiment 1: chunk concatenation. Tests whether grouping 2-3
topically-related FAQ answers into a single coarser chunk changes retrieval
quality, independently of baseline (not stacked with experiments 2 or 3).

This script builds the collection and scores it with retrieval_eval's shared
metrics; the collection-building and query logic themselves live in
src/chunking.py, since that config went on to win Phase 5 and is what
src/api.py actually serves in production — see src/chunking.py's module
docstring for the grouping method, its caveats, and the sub-answer citation
fix (that part only matters for generation quality, scored separately by
src/eval/answer_eval_chunking.py, not by this retrieval-only script).

Each eval query's gold label is remapped from its original answer_id to
whichever chunk now contains it (gold_id_fn below), since the KB granularity
changed — retrieval_eval.run_experiment() handles the remap, and
find_paraphrase_answer_ids() still operates on the original, un-remapped
answer_id (kept as original_answer_id on every record) so the paraphrase
subset means the same 6 queries as every other experiment.

Run: python -m src.eval.experiment_chunking
"""
import json
from pathlib import Path

from src.chunking import build_index, load_answer_id_to_chunk_id, make_retrieve_ids_fn
from src.eval.retrieval_eval import compute_metrics, find_paraphrase_answer_ids, print_markdown_table, run_experiment
from src.retrieve import get_client

EVAL_SET_PATH = Path("data/eval/eval_set.json")
RESULTS_PATH = Path("data/eval/retrieval_results_chunking.json")


def main() -> None:
    chunks = build_index()
    group_sizes = [len(c["answer_ids"]) for c in chunks]
    print(
        f"built {len(chunks)} chunks from {sum(group_sizes)} answers "
        f"(sizes: {sorted(set(group_sizes))}, counts {[group_sizes.count(s) for s in sorted(set(group_sizes))]})"
    )

    answer_id_to_chunk_id = load_answer_id_to_chunk_id()

    eval_rows = json.loads(EVAL_SET_PATH.read_text())
    paraphrase_answer_ids = find_paraphrase_answer_ids(eval_rows)

    records = run_experiment(
        eval_rows,
        retrieve_ids_fn=make_retrieve_ids_fn(),
        gold_id_fn=lambda original_answer_id: answer_id_to_chunk_id[original_answer_id],
    )
    get_client().close()

    aggregate = compute_metrics(records)
    paraphrase_records = [r for r in records if r["original_answer_id"] in paraphrase_answer_ids]
    paraphrase = compute_metrics(paraphrase_records)

    results = {
        "config": {"name": "chunk_concatenation", "num_chunks": len(chunks), "group_sizes": group_sizes},
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

    print_markdown_table(aggregate, paraphrase, label="Experiment 1: chunk concatenation (2-3 answers/chunk)")
    print(f"\nfull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()