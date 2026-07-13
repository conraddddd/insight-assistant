"""Phase 5 follow-up, take 2: re-run the winning-config answer-eval with a
per-sub-answer citation scheme, fixing the 25.9% hallucinated-citation rate
found on the first attempt.

The first attempt (data/eval/answer_results_chunking_v1_broken_citations.json)
retrieved and generated against kb_docs_concat using one id per whole chunk
(e.g. "[19]" for a chunk bundling 3 original answers). That collapsed the
1:1 mapping between "a citable fact" and "an id" the citation mechanism
depends on: when the model wanted to cite the *specific* sub-answer it used
within a chunk, it had no real sub-id to point to and invented a small
positional number instead ("[1]", "[2]"...) — confirmed by the fact that
hallucinated citations clustered overwhelmingly at small integers rather than
being scattered across the real chunk-id range (0-28). Faithfulness itself
barely moved (4.82 -> 4.76) — the underlying claims were still grounded in
retrieved text — so this was specifically a citation-hygiene failure, not a
factual one, which is exactly why faithfulness and hallucinated-citation-rate
are tracked as separate metrics rather than one blended score.

Fix: give each sub-answer within a chunk its own citable sub-id, built at
generation time — not by re-indexing kb_docs_concat, whose vectors/payloads
are untouched — as "<chunk_id><letter>" (chunk 19 with 3 members becomes
"19a", "19b", "19c"; experiment_chunking.build_subanswer_context). The
generation context lists every original answer on its own line under its own
sub-id, restoring the same 1:1 fact-to-id mapping the baseline (single-answer
-per-chunk) collection already has natively. generate.CONCAT_SYSTEM_PROMPT
and generate.cited_sub_ids() are the sub-id-aware counterparts of
SYSTEM_PROMPT/cited_ids() used for this path.

Retrieval itself is unchanged from the first attempt: still the top-5 chunks
from kb_docs_concat, still scored for retrieval_hit via whether the chunk
containing the gold answer_id was retrieved. Only what's shown to the
generator, and what counts as a valid citation, changed.

A second, independent bug surfaced during this same verification pass:
generate.ABSTAIN_SCORE_THRESHOLD (0.55) was calibrated on the baseline
collection's score distribution and doesn't transfer to kb_docs_concat, where
chunk embeddings blend 2-3 answers together and dilute cosine similarity for
any single one of them. One query ("What is your email newsletter about?")
scored 0.547 on its correct top-ranked chunk — just under 0.55 — and was
wrongly gated into an abstention despite the right answer being retrieved.
Fixed with a separate CONCAT_ABSTAIN_SCORE_THRESHOLD (0.50), calibrated the
same provisional, one-observed-gap way as the original: genuine-query top-1
scores over the 89-query eval set ranged 0.547-0.918, three known-irrelevant
probe queries scored 0.383-0.483.

Run: python -m src.eval.answer_eval_chunking
"""
import json
from pathlib import Path

from src.eval.answer_eval import (
    EVAL_SET_PATH,
    judge_completeness,
    judge_faithfulness,
    judge_relevance,
    print_markdown_table,
    summarize,
)
from src.chunking import build_subanswer_context, load_answer_id_to_chunk_id, make_retrieve_points_fn
from src.eval.retrieval_eval import find_paraphrase_answer_ids
from src.generate import CONCAT_ABSTAIN_SCORE_THRESHOLD, CONCAT_SYSTEM_PROMPT, cited_sub_ids, generate
from src.ingest import DATA_PATH, dedupe_and_split, load_faqs
from src.retrieve import RetrievedChunk, get_client

RESULTS_PATH = Path("data/eval/answer_results_chunking.json")


def evaluate_query(
    question: str,
    gold_answer_id: int,
    answer_by_id: dict[int, str],
    retrieve_points_fn,
    answer_id_to_chunk_id: dict[int, int],
) -> dict:
    points = retrieve_points_fn(question)
    retrieved_chunk_ids = [point.id for point in points]
    context_block, valid_sub_ids = build_subanswer_context(points, answer_by_id)

    # Lightweight RetrievedChunks purely so generate()'s abstention gate can
    # read chunks[0].score — the actual prompt content comes from
    # context_block, passed separately below.
    gate_chunks = [
        RetrievedChunk(answer_id=point.id, answer="", source_question="", score=point.score) for point in points
    ]
    draft = generate(
        question,
        gate_chunks,
        system_prompt=CONCAT_SYSTEM_PROMPT,
        context_block=context_block,
        abstain_threshold=CONCAT_ABSTAIN_SCORE_THRESHOLD,
    )

    cited = cited_sub_ids(draft)
    hallucinated = sorted(cited - valid_sub_ids)

    faithfulness = judge_faithfulness(draft, context_block)
    relevance = judge_relevance(question, draft)
    completeness = judge_completeness(answer_by_id[gold_answer_id], draft)

    gold_chunk_id = answer_id_to_chunk_id[gold_answer_id]

    return {
        "question": question,
        "gold_answer_id": gold_answer_id,
        "retrieved_ids": retrieved_chunk_ids,
        "cited_ids": sorted(cited),
        "hallucinated_citations": hallucinated,
        "retrieval_hit": gold_chunk_id in retrieved_chunk_ids,
        "faithfulness": faithfulness,
        "relevance": relevance,
        "completeness": completeness,
    }


def main() -> None:
    eval_rows = json.loads(EVAL_SET_PATH.read_text())
    paraphrase_answer_ids = find_paraphrase_answer_ids(eval_rows)

    faqs = load_faqs(DATA_PATH)
    kb_records, _ = dedupe_and_split(faqs)
    answer_by_id = {r["answer_id"]: r["answer"] for r in kb_records}

    answer_id_to_chunk_id = load_answer_id_to_chunk_id()
    retrieve_points_fn = make_retrieve_points_fn()

    records = []
    for i, row in enumerate(eval_rows, start=1):
        print(f"[{i}/{len(eval_rows)}] {row['question']}")
        records.append(
            evaluate_query(row["question"], row["answer_id"], answer_by_id, retrieve_points_fn, answer_id_to_chunk_id)
        )
    get_client().close()

    aggregate = summarize(records)
    paraphrase_records = [r for r in records if r["gold_answer_id"] in paraphrase_answer_ids]
    paraphrase = summarize(paraphrase_records)

    results = {
        "config": "chunk_concatenation with sub-answer citation ids (fix for v1's 25.9% hallucination rate)",
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

    print()
    print_markdown_table(aggregate, paraphrase, label="Winning config, fixed: chunk concatenation + sub-answer citation ids")
    print(f"\nfull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()