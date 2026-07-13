"""Phase 5, experiment 2: hybrid BM25 + vector retrieval, fused via
Reciprocal Rank Fusion (RRF). Tested independently against baseline (not
stacked with experiments 1 or 3) — same KB granularity and same baseline
kb_docs collection as retrieval_eval.py, only the retrieval algorithm changes.

Vector search and BM25 each retrieve their own top-20 candidates for a query;
RRF fuses the two ranked lists by

    fused_score(id) = sum over rankers r that surfaced id of  1 / (RRF_K + rank_r(id))

where rank_r(id) is id's 1-indexed position in ranker r's top-20 (an id
absent from a ranker's list contributes 0 from that ranker, not a penalty).
RRF_K = 60 is the standard constant from Cormack et al. (2009), chosen there
to keep any single rank-1 hit from dominating the fused score outright — we
inherit it rather than tune it, since tuning RRF_K is its own experiment this
project isn't running. The top-5 ids by fused score become the final result,
matching every other experiment's top_k.

BM25 (rank_bm25.BM25Okapi) runs over the same 86 KB answer texts used to
build the baseline collection, indexed in the same answer_id order (kb_records
is built by ingest.dedupe_and_split(), which assigns answer_id sequentially
by first-appearance order — so kb_records[i]["answer_id"] == i always, and
BM25's corpus index doubles as the answer_id with no separate lookup table).
Tokenization is a plain lowercase \\w+ split — no stemming/stopword removal,
since these are short single-sentence answers where a heavier NLP pipeline is
unlikely to matter and would just be another undefended design knob.

Run: python -m src.eval.experiment_hybrid
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.eval.retrieval_eval import TOP_K, compute_metrics, find_paraphrase_answer_ids, print_markdown_table, run_experiment
from src.ingest import DATA_PATH, dedupe_and_split, load_faqs
from src.retrieve import get_client, retrieve

EVAL_SET_PATH = Path("data/eval/eval_set.json")
RESULTS_PATH = Path("data/eval/retrieval_results_hybrid.json")
RRF_K = 60
FUSION_DEPTH = 20

TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def make_retrieve_ids_fn(bm25: BM25Okapi, top_k: int = TOP_K, fusion_depth: int = FUSION_DEPTH):
    def retrieve_ids(question: str) -> list[int]:
        vector_hits = retrieve(question, top_k=fusion_depth)
        vector_ranked_ids = [c.answer_id for c in vector_hits]

        bm25_scores = bm25.get_scores(tokenize(question))
        bm25_ranked_ids = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:fusion_depth]

        fused = defaultdict(float)
        for rank, answer_id in enumerate(vector_ranked_ids, start=1):
            fused[answer_id] += 1 / (RRF_K + rank)
        for rank, answer_id in enumerate(bm25_ranked_ids, start=1):
            fused[answer_id] += 1 / (RRF_K + rank)

        return [answer_id for answer_id, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]]

    return retrieve_ids


def main() -> None:
    faqs = load_faqs(DATA_PATH)
    kb_records, _ = dedupe_and_split(faqs)
    bm25 = BM25Okapi([tokenize(r["answer"]) for r in kb_records])

    eval_rows = json.loads(EVAL_SET_PATH.read_text())
    paraphrase_answer_ids = find_paraphrase_answer_ids(eval_rows)

    records = run_experiment(eval_rows, retrieve_ids_fn=make_retrieve_ids_fn(bm25))
    get_client().close()

    aggregate = compute_metrics(records)
    paraphrase_records = [r for r in records if r["original_answer_id"] in paraphrase_answer_ids]
    paraphrase = compute_metrics(paraphrase_records)

    results = {
        "config": {"name": "hybrid_bm25_vector_rrf", "rrf_k": RRF_K, "fusion_depth": FUSION_DEPTH},
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

    print_markdown_table(aggregate, paraphrase, label="Experiment 2: hybrid BM25 + vector (RRF)")
    print(f"\nfull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()