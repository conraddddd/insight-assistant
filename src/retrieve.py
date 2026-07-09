"""Phase 3 (retrieval half): embed a query and fetch the top-k nearest KB
answers from the local Qdrant collection built by ingest.py.

Run: python -m src.retrieve "How do I reset my password?"
"""
import sys
from dataclasses import dataclass

import ollama
from qdrant_client import QdrantClient

from src.config import settings


@dataclass
class RetrievedChunk:
    answer_id: int
    answer: str
    source_question: str
    score: float


_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Qdrant in embedded/local mode opens an on-disk DB and takes a file lock
    on it — only one process/client can hold that path open at a time. A
    single lazily-created module-level client (rather than a new QdrantClient
    per call) avoids re-opening the same path repeatedly within one process,
    which is what a FastAPI app doing this per-request would otherwise do."""
    global _client
    if _client is None:
        _client = QdrantClient(path=settings.qdrant_path)
    return _client


def embed_query(text: str) -> list[float]:
    response = ollama.embed(model=settings.embed_model, input=text)
    return response.embeddings[0]


def retrieve(query: str, top_k: int = settings.top_k) -> list[RetrievedChunk]:
    """Embed the query with the same model used to embed the KB (required —
    query and document vectors must come from the same embedding space) and
    fetch the top_k nearest answers by cosine similarity via query_points,
    qdrant-client 1.18's replacement for the removed search()/search_batch()."""
    vector = embed_query(query)
    client = get_client()
    response = client.query_points(
        collection_name=settings.collection_name,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        RetrievedChunk(
            answer_id=point.id,
            answer=point.payload["answer"],
            source_question=point.payload["source_question"],
            score=point.score,
        )
        for point in response.points
    ]


def main() -> None:
    query = " ".join(sys.argv[1:]) or "How do I reset my password?"
    for rank, hit in enumerate(retrieve(query), start=1):
        print(f"{rank}. [{hit.score:.3f}] (answer_id={hit.answer_id}) {hit.answer}")
    # Embedded Qdrant holds a file lock on qdrant_db/ for as long as the client
    # is open. Closing it explicitly here (rather than letting __del__ run
    # during interpreter shutdown) releases that lock cleanly and avoids a
    # spurious "Python is likely shutting down" exception from __del__.
    get_client().close()


if __name__ == "__main__":
    main()
