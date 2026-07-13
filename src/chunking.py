"""The chunk-concatenated knowledge base: groups 2-3 topically-related FAQ
answers into a single coarser retrieval unit. This is production code, not
just an eval artifact — it won Phase 5's retrieval experiment
(src/eval/experiment_chunking.py) and, after the sub-answer citation fix
below, the LLM-as-judge answer-quality eval too
(src/eval/answer_eval_chunking.py) — so src/api.py and src/generate.py's
production path serve from this collection, not the single-answer baseline
built by src/ingest.py.

"Topically related" is operationalized via cosine similarity in the same
mxbai-embed-large embedding space already used for retrieval, rather than an
arbitrary separate heuristic — faqs.json has no category field to group by.

Grouping method: greedy nearest-neighbor chaining. Starting from answer_id 0,
repeatedly append whichever not-yet-visited answer has the highest cosine
similarity to the *current chain end*, producing an ordering where every
adjacent pair is a close embedding match. The resulting chain is then sliced
into consecutive runs of 3 (86 answers -> 28 groups of 3 + 1 group of 2 = 29
chunks, down from the 86 single-answer baseline chunks).

Caveat, stated plainly: chaining only guarantees adjacent-pair similarity
within a group (item 1-2, and item 2-3), not group-wide coherence — item 1
and item 3 in a group of 3 are never directly compared, so a chunk can
occasionally bridge two moderately different topics through the middle item.
This is an approximation of true topical clustering, not exact clustering,
chosen for simplicity and determinism over a guarantee of intra-group
coherence.

Builds a new Qdrant collection (kb_docs_concat, same local qdrant_db path,
different collection name) so the baseline kb_docs collection is never
touched or overwritten.

Sub-answer citation ids ("<chunk_id><letter>", e.g. "19a", "19b", "19c") are
built at query time from the persisted payload (build_subanswer_context
below), not by re-indexing. They exist because citing a whole multi-answer
chunk under one blanket id destroyed the 1:1 fact-to-id mapping the citation
mechanism depends on — the model resolved that ambiguity by inventing a
small positional number ("[1]", "[2]"...) instead of reusing a real id, a
25.9% hallucinated-citation rate before this fix (see
data/eval/answer_results_chunking_v1_broken_citations.json and
generate.CONCAT_SYSTEM_PROMPT).
"""
from qdrant_client.http.models import PointStruct

import numpy as np

from src.ingest import DATA_PATH, build_collection, dedupe_and_split, embed_texts, load_faqs
from src.retrieve import embed_query, get_client

COLLECTION_NAME = "kb_docs_concat"
GROUP_SIZE = 3
SUBID_LETTERS = "abcdefghijklmnopqrstuvwxyz"
TOP_K = 5


def build_chain(embeddings: list[list[float]]) -> list[int]:
    vectors = np.array(embeddings)
    normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = normed @ normed.T

    n = len(embeddings)
    visited = [False] * n
    chain = [0]
    visited[0] = True
    for _ in range(n - 1):
        current = chain[-1]
        next_id = max((j for j in range(n) if not visited[j]), key=lambda j: similarity[current][j])
        chain.append(next_id)
        visited[next_id] = True
    return chain


def slice_into_groups(chain: list[int], size: int = GROUP_SIZE) -> list[list[int]]:
    return [chain[i : i + size] for i in range(0, len(chain), size)]


def build_chunks(kb_records: list[dict], groups: list[list[int]]) -> list[dict]:
    answer_by_id = {r["answer_id"]: r["answer"] for r in kb_records}
    return [
        {
            "chunk_id": chunk_id,
            "answer_ids": group,
            "text": "\n\n".join(answer_by_id[aid] for aid in group),
        }
        for chunk_id, group in enumerate(groups)
    ]


def index_chunks(chunks: list[dict]) -> None:
    embeddings = embed_texts([c["text"] for c in chunks])
    client = get_client()
    build_collection(client, collection_name=COLLECTION_NAME)
    points = [
        PointStruct(
            id=c["chunk_id"],
            vector=vector,
            payload={"answer_ids": c["answer_ids"], "text": c["text"]},
        )
        for c, vector in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def build_index() -> list[dict]:
    """Full build: load faqs, embed, chain, group, index. Returns the built
    chunk records so callers that want build-time stats (e.g. group size
    distribution) don't need to recompute anything."""
    faqs = load_faqs(DATA_PATH)
    kb_records, _ = dedupe_and_split(faqs)
    kb_embeddings = embed_texts([r["answer"] for r in kb_records])

    chain = build_chain(kb_embeddings)
    groups = slice_into_groups(chain)
    chunks = build_chunks(kb_records, groups)
    index_chunks(chunks)
    return chunks


def load_answer_id_to_chunk_id() -> dict[int, int]:
    """Reads the mapping directly from kb_docs_concat's stored payloads
    (source of truth = what's actually indexed) rather than recomputing the
    chain/groups."""
    client = get_client()
    points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=1000, with_payload=True)
    return {answer_id: point.id for point in points for answer_id in point.payload["answer_ids"]}


def retrieve_ids(question: str, top_k: int = TOP_K) -> list[int]:
    vector = embed_query(question)
    response = get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        with_payload=False,
    )
    return [point.id for point in response.points]


def make_retrieve_ids_fn(top_k: int = TOP_K):
    return lambda question: retrieve_ids(question, top_k)


def retrieve_points(question: str, top_k: int = TOP_K):
    """Raw Qdrant points (id + payload), not RetrievedChunk objects — needed
    for the sub-answer citation scheme, which expands each chunk's individual
    answer_ids into per-fact sub-ids rather than one blanket id per chunk
    (see build_subanswer_context below)."""
    vector = embed_query(question)
    response = get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return response.points


def make_retrieve_points_fn(top_k: int = TOP_K):
    return lambda question: retrieve_points(question, top_k)


def sub_ids_for(chunk_id: int, answer_ids: list[int]) -> list[str]:
    return [f"{chunk_id}{SUBID_LETTERS[i]}" for i in range(len(answer_ids))]


def build_subanswer_context(points, answer_by_id: dict[int, str]) -> tuple[str, set[str]]:
    """Expands retrieved chunks into one line per original answer, each under
    its own sub-id ("19a", "19b", ...) instead of one line per chunk under a
    single blanket id — this is what restores a real, precise, citable id for
    every individual fact shown to the generator. Returns the context block
    text plus the full set of valid sub-ids shown, so the caller can check
    every cited sub-id against ids that were *actually shown* rather than
    trusting the model not to invent one."""
    lines = []
    valid_sub_ids = set()
    for point in points:
        answer_ids = point.payload["answer_ids"]
        for sub_id, answer_id in zip(sub_ids_for(point.id, answer_ids), answer_ids):
            lines.append(f"[{sub_id}] {answer_by_id[answer_id]}")
            valid_sub_ids.add(sub_id)
    return "\n".join(lines), valid_sub_ids


def load_answer_by_id() -> dict[int, str]:
    """The original single-FAQ answer texts, keyed by their baseline
    answer_id — needed alongside a chunk_id/answer_ids payload to resolve
    "which exact text does sub-id 19b refer to." Deterministic given
    faqs.json, so safe to call once at process startup (src.api)."""
    faqs = load_faqs(DATA_PATH)
    kb_records, _ = dedupe_and_split(faqs)
    return {r["answer_id"]: r["answer"] for r in kb_records}