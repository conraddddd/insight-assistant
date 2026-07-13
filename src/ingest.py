"""Phase 2: load faqs.json, dedupe, split into KB (answers) vs eval set (questions),
embed the KB with mxbai-embed-large, and index it into a local Qdrant collection.

Run: python -m src.ingest
"""
import json
from pathlib import Path

import ollama
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from src.config import settings

DATA_PATH = Path("data/faqs.json")
EVAL_SET_PATH = Path("data/eval/eval_set.json")

EMBED_DIM = 1024  # mxbai-embed-large output size — see note in embed_answers()


def load_faqs(path: Path) -> list[dict]:
    """faqs.json is JSON Lines (one {question, answer} object per line), not a
    single JSON array — json.load() on the whole file raises JSONDecodeError."""
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dedupe_and_split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into KB answers and eval questions.

    The source data has 200 rows but only 89 unique (question, answer) pairs —
    111 rows are exact duplicates. We dedupe on the exact pair first so the
    index doesn't end up with repeated identical vectors (which would inflate
    recall@k by giving every query multiple "correct" hits) and eval doesn't
    double-count the same question.

    KB and eval set are a *column* split, not a row holdout: every unique
    answer goes into the KB, every unique question becomes an eval query.
    There's no training step here, so the only leakage that matters is never
    embedding question text into the index — which this split guarantees by
    construction, since questions only ever end up in the eval set.

    A few answers are shared by two differently-worded questions (paraphrases,
    e.g. "out of stock" vs "currently out of stock"). We keep both questions as
    separate eval rows pointing at the same answer_id — that's a legitimate
    free test of retrieval robustness to phrasing, not a duplicate to collapse.
    """
    seen_pairs = set()
    answer_to_id: dict[str, int] = {}
    kb_records: list[dict] = []
    eval_records: list[dict] = []

    for r in records:
        pair = (r["question"], r["answer"])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        answer = r["answer"]
        if answer not in answer_to_id:
            answer_id = len(kb_records)
            answer_to_id[answer] = answer_id
            kb_records.append(
                {"answer_id": answer_id, "answer": answer, "source_question": r["question"]}
            )

        eval_records.append({"question": r["question"], "answer_id": answer_to_id[answer]})

    return kb_records, eval_records


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batching all texts into a single ollama.embed() call (rather than one
    call per text) cuts round-trips to the local Ollama server. Shared by
    embed_answers() below and by Phase 5's chunk-concatenation experiment,
    which embeds concatenated multi-answer chunk text through the same path."""
    response = ollama.embed(model=settings.embed_model, input=texts)
    embeddings = response.embeddings
    assert len(embeddings[0]) == EMBED_DIM, (
        f"expected {EMBED_DIM}-dim vectors from {settings.embed_model}, got {len(embeddings[0])}"
    )
    return embeddings


def embed_answers(kb_records: list[dict]) -> list[list[float]]:
    """One embedding call for all KB records. Each answer (91-219 chars) is
    already well below any sensible chunk size, so there's no sub-splitting
    here — one answer is one chunk is one vector."""
    return embed_texts([r["answer"] for r in kb_records])


def build_collection(client: QdrantClient, collection_name: str = settings.collection_name) -> None:
    """(Re)create the collection. VectorParams needs the embedding dimension
    (1024 for mxbai-embed-large) and a distance metric — cosine, since
    mxbai-embed-large is trained/normalized for cosine similarity, not
    Euclidean or dot product. Collection name is parameterized so Phase 5
    experiments can build alternate collections (e.g. kb_docs_concat) in the
    same local qdrant_db path without touching the baseline kb_docs
    collection. delete+create (rather than a bare create) makes this
    idempotent — re-running ingest during development just rebuilds."""
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )


def upsert_points(client: QdrantClient, kb_records: list[dict], embeddings: list[list[float]]) -> None:
    """Payload carries what generate.py will need to write a grounded, cited
    reply (the answer text itself) plus source_question, kept purely as debug
    metadata so a human can trace a retrieved chunk back to its original FAQ
    without cross-referencing faqs.json — it's never embedded or used for
    matching. The point id *is* the answer_id (int), so eval_set.json's
    answer_id lines up directly with Qdrant point ids with no extra lookup
    table."""
    points = [
        PointStruct(
            id=r["answer_id"],
            vector=vector,
            payload={"answer": r["answer"], "source_question": r["source_question"]},
        )
        for r, vector in zip(kb_records, embeddings)
    ]
    client.upsert(collection_name=settings.collection_name, points=points)


def main() -> None:
    records = load_faqs(DATA_PATH)
    kb_records, eval_records = dedupe_and_split(records)
    print(f"loaded {len(records)} rows -> {len(kb_records)} KB answers, {len(eval_records)} eval questions")

    embeddings = embed_answers(kb_records)

    client = QdrantClient(path=settings.qdrant_path)
    build_collection(client)
    upsert_points(client, kb_records, embeddings)
    print(f"indexed {len(kb_records)} answers into '{settings.collection_name}' at {settings.qdrant_path}")

    EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_SET_PATH.write_text(json.dumps(eval_records, indent=2))
    print(f"wrote {len(eval_records)} eval rows to {EVAL_SET_PATH}")


if __name__ == "__main__":
    main()
