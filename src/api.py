"""Phase 3 (serving): FastAPI wrapper around retrieve() + generate().

Run: uvicorn src.api:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from src.config import settings
from src.generate import cited_ids, generate
from src.retrieve import get_client, retrieve


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
    answer_id: int
    answer: str
    score: float


class DraftResponse(BaseModel):
    draft: str
    citations: list[Citation]


@app.post("/draft", response_model=DraftResponse)
def draft(request: DraftRequest) -> DraftResponse:
    chunks = retrieve(request.query, top_k=request.top_k)
    text = generate(request.query, chunks)

    # Only chunks the model actually cited (per the [id] bracket pattern) are
    # returned as citations — a chunk retrieve() surfaced but generate()
    # didn't use isn't part of the grounding for this particular reply.
    ids = cited_ids(text)
    citations = [
        Citation(answer_id=c.answer_id, answer=c.answer, score=c.score)
        for c in chunks
        if c.answer_id in ids
    ]
    return DraftResponse(draft=text, citations=citations)
