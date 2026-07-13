"""Serving path: FastAPI wrapper around the production retrieval+generation
pipeline. Serves from the chunk-concatenated collection (src/chunking.py),
the config that won Phase 5's retrieval experiment and, after the sub-answer
citation fix, the LLM-as-judge answer-quality eval too — not the single-
answer-per-chunk baseline src/ingest.py builds, which is now purely an eval
reference point (data/eval/retrieval_results.json / answer_results.json).

Run: uvicorn src.api:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from src.chunking import build_subanswer_context, load_answer_by_id, retrieve_points, sub_ids_for
from src.config import settings
from src.generate import CONCAT_ABSTAIN_SCORE_THRESHOLD, CONCAT_SYSTEM_PROMPT, cited_sub_ids, generate
from src.retrieve import RetrievedChunk, get_client

# Deterministic given faqs.json — safe to compute once at import time rather
# than per-request (see src.chunking.load_answer_by_id).
ANSWER_BY_ID = load_answer_by_id()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the embedded Qdrant client once at startup and hold it for the
    life of the process, rather than lazily on first request — and close it
    explicitly on shutdown to release the on-disk file lock deterministically
    instead of relying on __del__ during interpreter teardown (see the note
    in retrieve.get_client())."""
    get_client()
    yield
    get_client().close()


app = FastAPI(title="Insight", lifespan=lifespan)


class DraftRequest(BaseModel):
    query: str
    top_k: int = settings.top_k


class Citation(BaseModel):
    sub_id: str
    answer: str
    chunk_id: int
    score: float


class DraftResponse(BaseModel):
    draft: str
    citations: list[Citation]


@app.post("/draft", response_model=DraftResponse)
def draft(request: DraftRequest) -> DraftResponse:
    points = retrieve_points(request.query, top_k=request.top_k)
    context_block, valid_sub_ids = build_subanswer_context(points, ANSWER_BY_ID)

    # Lightweight RetrievedChunks purely so generate()'s abstention gate can
    # read chunks[0].score — the actual prompt content comes from
    # context_block, passed separately below.
    gate_chunks = [
        RetrievedChunk(answer_id=p.id, answer="", source_question="", score=p.score) for p in points
    ]
    text = generate(
        request.query,
        gate_chunks,
        system_prompt=CONCAT_SYSTEM_PROMPT,
        context_block=context_block,
        abstain_threshold=CONCAT_ABSTAIN_SCORE_THRESHOLD,
    )

    # Only sub-ids the model actually cited, and only if they were actually
    # shown to it — a chunk retrieve_points() surfaced but generate() didn't
    # cite isn't part of the grounding for this particular reply, and a
    # cited sub-id outside valid_sub_ids would be a hallucination (excluded,
    # not surfaced to the caller as if it were a real source).
    cited = cited_sub_ids(text) & valid_sub_ids
    sub_id_info = {
        sub_id: {"chunk_id": p.id, "score": p.score, "answer_id": answer_id}
        for p in points
        for sub_id, answer_id in zip(sub_ids_for(p.id, p.payload["answer_ids"]), p.payload["answer_ids"])
    }
    citations = [
        Citation(
            sub_id=sub_id,
            answer=ANSWER_BY_ID[sub_id_info[sub_id]["answer_id"]],
            chunk_id=sub_id_info[sub_id]["chunk_id"],
            score=sub_id_info[sub_id]["score"],
        )
        for sub_id in sorted(cited)
    ]
    return DraftResponse(draft=text, citations=citations)