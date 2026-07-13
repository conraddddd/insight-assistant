"""Phase 4b: LLM-as-judge evaluation of drafted replies, over the same 89-query
eval set used by retrieval_eval.py. This runs the full production pipeline
(retrieve -> generate) for each query, then scores each draft on three axes.

Judge model: qwen2.5:7b, deliberately a different model family from the
generator (llama3). This guards against self-preference bias — a documented
effect where a model rates outputs written in its own style/phrasing more
favorably than an independent judge would. Qwen (Alibaba) and Llama (Meta)
come from different training data and RLHF recipes, so a shared stylistic
blind spot between generator and judge is much less likely than if llama3
judged its own drafts.

Three metrics, three separate judge calls per query, each seeing only the
inputs that metric needs — this is deliberate, not an economy measure:
  - Faithfulness sees {draft, retrieved context}. Not the query or gold
    answer, so the judge can't reward "answered the right question," only
    "didn't assert anything the context doesn't support."
  - Relevance sees {draft, query}. Not the gold answer or context, so the
    judge can't reward matching the reference's wording, only whether the
    draft addresses what the customer actually asked.
  - Completeness sees {draft, gold answer}. Not the query or context, so the
    judge purely measures information coverage against the reference, not
    phrasing quality or topical fit (those are relevance's job).
Each rubric is anchored 1-5 (see FAITHFULNESS_RUBRIC / RELEVANCE_RUBRIC /
COMPLETENESS_RUBRIC below). Faithfulness splits its two failure levels by
severity: level 1 is reserved for actively contradicting the context (worse
in a support setting — telling a customer something false), level 2 for
fabricating unsupported-but-not-contradictory additions.

On this eval set every query has a real gold answer by construction (that's
how eval_set.json was built from the FAQ Q/A pairs), so an abstention here is
always a genuine miss, never an appropriate deflection. An abstaining draft
factually addresses nothing and covers nothing, so it lands at relevance=1 /
completeness=1 on the rubrics as written — no special-casing needed. Paired
with vacuously-high faithfulness (an abstention makes no claims, so nothing
in it is unsupported), that specific combination — faithfulness 5, relevance
1, completeness 1 — is the diagnostic signature of the Phase 3 score gate
(ABSTAIN_SCORE_THRESHOLD in generate.py, still a provisional placeholder)
firing on a query it shouldn't have.

Structured output: each judge call requests JSON matching
{"score": int, "rationale": str} via Ollama's schema-constrained format
parameter. That constrains the *shape* of the output but not whether the
content is sensible — a schema-valid response can still report a score
outside 1-5, or fail to parse if the model wraps it in extra text despite the
schema. Any JSON parse failure, missing field, or out-of-range score is
treated as malformed: logged with the raw response, excluded from that
metric's mean, and counted, so a high malformed rate is visible rather than
silently absorbed into the average.

Hallucinated citation rate is NOT judged by the LLM — it's a deterministic
set-membership check (does generate.py's cited_ids(draft) contain an
answer_id that wasn't even in retrieve()'s output for that query?) computed
directly in code. LLM judgment is reserved for genuinely subjective axes;
this one is enumerable, so asking a model to eyeball it would just add noise.

Completeness is reported two ways, because it's a joint retrieval+generation
signal: a retrieval miss (gold answer not in the retrieved top-k) makes a
complete draft structurally impossible regardless of how good the generator
is. So completeness is aggregated once over all 89 queries, and again
restricted to the subset where retrieval actually succeeded
(gold_answer_id in retrieved_ids, read from retrieval_eval's own per-query
`rank` field in data/eval/retrieval_results.json). The gap between the two is
the retrieval-induced ceiling on completeness — named explicitly in the
printed output rather than left for the reader to notice.

A paraphrase-pair subset breakdown (n=6, the same 3 answer_ids each shared by
two differently-worded questions used in retrieval_eval.py) is reported
alongside the aggregate, marked illustrative only for the same reason: n=6 is
nowhere near large enough to support a real conclusion on its own.

Run: python -m src.eval.answer_eval
"""
import json
from pathlib import Path

import ollama

from src.eval.retrieval_eval import TOP_K, find_paraphrase_answer_ids
from src.generate import build_context_block, cited_ids, generate
from src.ingest import DATA_PATH, dedupe_and_split, load_faqs
from src.retrieve import get_client, retrieve

JUDGE_MODEL = "qwen2.5:7b"
EVAL_SET_PATH = Path("data/eval/eval_set.json")
RETRIEVAL_RESULTS_PATH = Path("data/eval/retrieval_results.json")
RESULTS_PATH = Path("data/eval/answer_results.json")

JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
}

FAITHFULNESS_RUBRIC = """\
5 - Every claim in the draft is directly supported by the retrieved context.
4 - Fully supported; only trivial rephrasing/connective wording beyond the context.
3 - Mostly supported, with one minor unsupported detail or slight overgeneralization.
2 - Fabricates unsupported claims - adds content beyond what the context supports.
1 - Actively contradicts the retrieved context."""

RELEVANCE_RUBRIC = """\
5 - Directly and fully addresses the customer's specific question; nothing extraneous or off-topic.
4 - Addresses the question well, with minor extraneous content or slightly generic framing.
3 - Partially addresses the question, or is somewhat generic/vague relative to what was asked.
2 - Only tangentially related - addresses a nearby but different topic than what was asked.
1 - Does not address the customer's question at all."""

COMPLETENESS_RUBRIC = """\
5 - Covers all key information present in the reference answer; nothing material missing.
4 - Covers nearly all key information; only supplementary phrasing or elaboration is omitted.
3 - Omits at least one actionable detail from the reference answer (a step, condition, alternative, or caveat).
2 - Covers only a fragment of the reference answer's content; multiple key details missing.
1 - Omits the substance of the reference answer entirely."""


def _judge_system_prompt(metric_name: str, rubric: str) -> str:
    return (
        f"You are an evaluator scoring a customer-support email draft for "
        f"{metric_name}.\n\nScore 1-5 using this rubric:\n{rubric}\n\n"
        f'Respond with JSON only: {{"score": <1-5 integer>, "rationale": "<one sentence>"}}'
    )


FAITHFULNESS_SYSTEM = _judge_system_prompt(
    "FAITHFULNESS — whether it only asserts things the retrieved context actually supports",
    FAITHFULNESS_RUBRIC,
) + (
    "\n\nIf the draft makes no substantive factual claims at all (for example, it "
    "politely declines to answer), that is vacuously faithful — score 5, since "
    "nothing it asserts is unsupported. Not addressing the context is not the "
    "same as contradicting it."
)
RELEVANCE_SYSTEM = _judge_system_prompt(
    "RELEVANCE — whether it addresses the customer's specific question",
    RELEVANCE_RUBRIC,
)
COMPLETENESS_SYSTEM = _judge_system_prompt(
    "COMPLETENESS — whether it covers the key information in the reference answer",
    COMPLETENESS_RUBRIC,
)


def call_judge(system_prompt: str, user_prompt: str) -> dict | None:
    response = ollama.chat(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format=JUDGE_RESPONSE_SCHEMA,
        options={"temperature": 0.0},
    )
    raw = response.message.content
    try:
        parsed = json.loads(raw)
        score = int(parsed["score"])
        rationale = str(parsed["rationale"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        print(f"  [malformed judge output, excluded] raw={raw!r}")
        return None
    if score not in (1, 2, 3, 4, 5):
        print(f"  [malformed judge output, excluded] score out of range: {raw!r}")
        return None
    return {"score": score, "rationale": rationale}


def judge_faithfulness(draft: str, context_block: str) -> dict | None:
    user = f"Context:\n{context_block}\n\nDraft:\n{draft}"
    return call_judge(FAITHFULNESS_SYSTEM, user)


def judge_relevance(query: str, draft: str) -> dict | None:
    user = f"Customer question:\n{query}\n\nDraft:\n{draft}"
    return call_judge(RELEVANCE_SYSTEM, user)


def judge_completeness(gold_answer: str, draft: str) -> dict | None:
    user = f"Reference answer:\n{gold_answer}\n\nDraft:\n{draft}"
    return call_judge(COMPLETENESS_SYSTEM, user)


def evaluate_query(
    question: str,
    gold_answer_id: int,
    answer_by_id: dict[int, str],
    retrieve_fn=lambda q: retrieve(q, top_k=TOP_K),
    is_hit_fn=lambda gold_answer_id, retrieved_ids: gold_answer_id in retrieved_ids,
) -> dict:
    """retrieve_fn and is_hit_fn default to the baseline retrieve() and a
    plain membership check, but Phase 5's winning-config rerun
    (answer_eval_chunking.py) substitutes both: a retrieve_fn querying the
    kb_docs_concat collection, and an is_hit_fn that checks whether the chunk
    containing gold_answer_id was retrieved, since the retrieval unit changed
    from single answers to multi-answer chunks."""
    chunks = retrieve_fn(question)
    draft = generate(question, chunks)

    retrieved_ids = [c.answer_id for c in chunks]
    cited = cited_ids(draft)
    hallucinated = sorted(cited - set(retrieved_ids))

    faithfulness = judge_faithfulness(draft, build_context_block(chunks))
    relevance = judge_relevance(question, draft)
    completeness = judge_completeness(answer_by_id[gold_answer_id], draft)

    return {
        "question": question,
        "gold_answer_id": gold_answer_id,
        "retrieved_ids": retrieved_ids,
        "cited_ids": sorted(cited),
        "hallucinated_citations": hallucinated,
        "retrieval_hit": is_hit_fn(gold_answer_id, retrieved_ids),
        "faithfulness": faithfulness,
        "relevance": relevance,
        "completeness": completeness,
    }


def aggregate_metric(records: list[dict], key: str) -> dict:
    scores = [r[key]["score"] for r in records if r[key] is not None]
    malformed = sum(1 for r in records if r[key] is None)
    return {
        "mean": sum(scores) / len(scores) if scores else None,
        "n": len(scores),
        "malformed_excluded": malformed,
    }


def hallucination_rate(records: list[dict]) -> dict:
    total_citations = sum(len(r["cited_ids"]) for r in records)
    total_hallucinated = sum(len(r["hallucinated_citations"]) for r in records)
    queries_with_hallucination = sum(1 for r in records if r["hallucinated_citations"])
    return {
        "rate": total_hallucinated / total_citations if total_citations else 0.0,
        "hallucinated_citations": total_hallucinated,
        "total_citations": total_citations,
        "queries_with_hallucination": queries_with_hallucination,
    }


def summarize(records: list[dict]) -> dict:
    hit_records = [r for r in records if r["retrieval_hit"]]
    return {
        "n": len(records),
        "faithfulness": aggregate_metric(records, "faithfulness"),
        "relevance": aggregate_metric(records, "relevance"),
        "completeness": aggregate_metric(records, "completeness"),
        "completeness_on_retrieval_hits": aggregate_metric(hit_records, "completeness") | {"n_hit_queries": len(hit_records)},
        "hallucination": hallucination_rate(records),
    }


def print_markdown_table(aggregate: dict, paraphrase: dict, label: str = "Baseline") -> None:
    def mean_str(block: dict) -> str:
        return f"{block['mean']:.3f}" if block["mean"] is not None else "n/a"

    print(f"### {label}")
    print(f"| Metric | All queries (n={aggregate['n']}) | Paraphrase-pair subset (n={paraphrase['n']}, illustrative) |")
    print("|---|---|---|")
    print(f"| Faithfulness (mean 1-5) | {mean_str(aggregate['faithfulness'])} | {mean_str(paraphrase['faithfulness'])} |")
    print(f"| Relevance (mean 1-5) | {mean_str(aggregate['relevance'])} | {mean_str(paraphrase['relevance'])} |")
    print(f"| Completeness, all (mean 1-5) | {mean_str(aggregate['completeness'])} | {mean_str(paraphrase['completeness'])} |")
    print(
        f"| Completeness, retrieval-hit subset (mean 1-5, n={aggregate['completeness_on_retrieval_hits']['n_hit_queries']}) "
        f"| {mean_str(aggregate['completeness_on_retrieval_hits'])} | {mean_str(paraphrase['completeness_on_retrieval_hits'])} |"
    )
    print(f"| Hallucinated citation rate | {aggregate['hallucination']['rate']:.1%} | {paraphrase['hallucination']['rate']:.1%} |")
    print()
    all_c, hit_c = aggregate["completeness"]["mean"], aggregate["completeness_on_retrieval_hits"]["mean"]
    if all_c is not None and hit_c is not None:
        print(
            f"Retrieval-induced completeness ceiling: {hit_c:.3f} on the "
            f"{aggregate['completeness_on_retrieval_hits']['n_hit_queries']} queries where retrieval actually "
            f"surfaced the gold answer, vs. {all_c:.3f} overall — the "
            f"{hit_c - all_c:.3f}-point gap is generation quality bounded by "
            f"retrieval misses, not a generation-side failure."
        )
    for metric in ("faithfulness", "relevance", "completeness"):
        malformed = aggregate[metric]["malformed_excluded"]
        if malformed:
            print(f"{metric}: {malformed} malformed judge output(s) excluded (see stderr-style log above).")


def main() -> None:
    eval_rows = json.loads(EVAL_SET_PATH.read_text())
    paraphrase_answer_ids = find_paraphrase_answer_ids(eval_rows)

    faqs = load_faqs(DATA_PATH)
    kb_records, _ = dedupe_and_split(faqs)
    answer_by_id = {r["answer_id"]: r["answer"] for r in kb_records}

    records = []
    for i, row in enumerate(eval_rows, start=1):
        print(f"[{i}/{len(eval_rows)}] {row['question']}")
        records.append(evaluate_query(row["question"], row["answer_id"], answer_by_id))
    get_client().close()

    aggregate = summarize(records)
    paraphrase_records = [r for r in records if r["gold_answer_id"] in paraphrase_answer_ids]
    paraphrase = summarize(paraphrase_records)

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

    print()
    print_markdown_table(aggregate, paraphrase)
    print(f"\nfull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()