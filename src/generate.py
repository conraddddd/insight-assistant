"""Phase 3 (generation half): take a customer query and the RetrievedChunks
retrieve.py found for it, and draft a grounded, cited email reply via llama3.

Run: python -m src.generate "How do I reset my password?"
"""
import re
import sys

import ollama

from src.config import settings
from src.retrieve import RetrievedChunk, get_client, retrieve

SYSTEM_PROMPT = """\
You are a customer support agent replying to a customer's email. Write your
reply using ONLY the information given in the CONTEXT section of the user
message below — do not use outside knowledge, and do not invent policies,
numbers, or facts that are not present in the context.

Rules:
1. Every factual claim in your reply must be supported by at least one
   context entry. Immediately after such a sentence, cite the entry using its
   id in square brackets, e.g. "You can reset your password from the login
   page [3]." Cite only ids that actually appear in the CONTEXT section —
   never invent an id.
2. Judge whether the context actually answers the customer's specific
   question — the mere presence of context entries does not mean they are
   relevant. If none of them directly address what the customer asked (for
   example, the customer asks about the weather and the context is only about
   restock notifications), do not stretch an unrelated entry into an answer.
   Instead say plainly that you don't have that information, and suggest the
   customer contact support directly for further help. Do not attach a
   citation to that sentence.
3. Write the reply as a real email: a brief greeting, a short body in plain
   prose paragraphs, and a sign-off. No bullet points, no headers.
4. Keep the tone warm, concise, and professional.
5. Output ONLY the email itself. No preamble like "Certainly! Here's a
   reply:", no commentary, no explanation of what you did — the first
   character of your output must be the first character of the email.
"""

CITATION_RE = re.compile(r"\[(\d+)\]")

# Sub-answer citation variant for Phase 5's chunk-concatenation experiment.
# When a retrieval unit bundles multiple original answers under one id (a
# "chunk"), citing the bare chunk id no longer identifies which specific
# fact within it supports a given sentence — testing showed the model
# resolves that ambiguity by inventing a small positional number ("[1]",
# "[2]"...) instead of reusing a real id, a 25.9% hallucinated-citation rate
# (see data/eval/answer_results_chunking_v1_broken_citations.json). The fix
# is to give every original answer within a chunk its own sub-id
# ("<chunk_id><letter>", e.g. "19a", "19b", "19c" — built in
# experiment_chunking.build_subanswer_context, not by re-indexing), so every
# individual fact has a real, precise, citable id again.
CONCAT_SYSTEM_PROMPT = """\
You are a customer support agent replying to a customer's email. Write your
reply using ONLY the information given in the CONTEXT section of the user
message below — do not use outside knowledge, and do not invent policies,
numbers, or facts that are not present in the context.

Rules:
1. Every factual claim in your reply must be supported by at least one
   context entry. Immediately after such a sentence, cite the entry using its
   exact id in square brackets. Context entries here are labeled with sub-ids
   like "19a", "19b", "19c" — a number followed by a letter — because each
   context entry groups multiple original answers together, and the letter
   identifies which specific one you're citing. Cite only sub-ids that appear
   EXACTLY as shown in the CONTEXT section (e.g. "[19a]") — never a bare
   number without its letter, and never an id you haven't seen.
2. Judge whether the context actually answers the customer's specific
   question — the mere presence of context entries does not mean they are
   relevant. If none of them directly address what the customer asked (for
   example, the customer asks about the weather and the context is only about
   restock notifications), do not stretch an unrelated entry into an answer.
   Instead say plainly that you don't have that information, and suggest the
   customer contact support directly for further help. Do not attach a
   citation to that sentence.
3. Write the reply as a real email: a brief greeting, a short body in plain
   prose paragraphs, and a sign-off. No bullet points, no headers.
4. Keep the tone warm, concise, and professional.
5. Output ONLY the email itself. No preamble like "Certainly! Here's a
   reply:", no commentary, no explanation of what you did — the first
   character of your output must be the first character of the email.
"""

CITATION_SUBID_RE = re.compile(r"\[(\d+[a-z])\]")

# Deterministic backstop below the prompt-level abstention rule: llama3 has
# shown it will rationalize a connection between the query and clearly
# unrelated chunks (e.g. "what's the weather" -> restock-notification answers
# at cosine ~0.45-0.48) rather than recognize they're irrelevant. Below this
# score there is no reliable match in the KB, so we skip the LLM call
# entirely rather than trust it to self-abstain. 0.55 is a provisional
# placeholder picked from one observed gap (0.44-0.48 for an unrelated query
# vs 0.83-0.87 for a genuine match) - it is NOT a tuned value. Revisit once
# Phase 4/5 have real precision/recall numbers to pick a threshold from.
ABSTAIN_SCORE_THRESHOLD = 0.55

# The concat collection needs its own threshold, not a reused 0.55: chunk
# embeddings blend 2-3 answers together, which dilutes cosine similarity for
# any single one of them, shifting the whole score distribution down.
# Verified empirically over the 89-query eval set plus 3 known-irrelevant
# probes: genuine-query top-1 scores ranged 0.547-0.918 (median 0.800),
# irrelevant probes scored 0.383-0.483. 0.50 splits that gap — same
# provisional, one-observed-gap methodology as ABSTAIN_SCORE_THRESHOLD above,
# not a tuned value. Using 0.55 here would have wrongly abstained on a real,
# answerable query (observed: "What is your email newsletter about?" scored
# 0.547 on its correct top-ranked chunk).
CONCAT_ABSTAIN_SCORE_THRESHOLD = 0.50

ABSTENTION_MESSAGE = """\
Dear customer,

Thank you for reaching out. Unfortunately, we don't have information on \
that in our records. Please contact our support team directly so they can \
look into this for you.

Best regards,
Customer Support"""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    return "\n".join(f"[{chunk.answer_id}] {chunk.answer}" for chunk in chunks)


def _build_user_message(query: str, context_block: str) -> str:
    return f'Customer email:\n"""\n{query}\n"""\n\nContext:\n{context_block}'


def build_user_message(query: str, chunks: list[RetrievedChunk]) -> str:
    return _build_user_message(query, build_context_block(chunks))


def generate(
    query: str,
    chunks: list[RetrievedChunk],
    system_prompt: str = SYSTEM_PROMPT,
    context_block: str | None = None,
    abstain_threshold: float = ABSTAIN_SCORE_THRESHOLD,
) -> str:
    """Low temperature (0.2) since this is grounded generation, not creative
    writing — we want the model to consistently follow the citation/abstention
    rules rather than vary its phrasing run to run.

    system_prompt/context_block let a caller override the default
    single-answer-per-chunk prompt and context formatting — used by the
    chunk-concatenation production path to swap in CONCAT_SYSTEM_PROMPT and a
    sub-answer-expanded context block, while still using chunks[0].score for
    the abstention gate below (chunks only needs valid .score values for
    that; the actual text shown to the model comes from context_block when
    given). abstain_threshold likewise lets that same caller supply
    CONCAT_ABSTAIN_SCORE_THRESHOLD instead of the baseline's calibration."""
    if not chunks or chunks[0].score < abstain_threshold:
        return ABSTENTION_MESSAGE

    block = context_block if context_block is not None else build_context_block(chunks)
    response = ollama.chat(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_user_message(query, block)},
        ],
        options={"temperature": 0.2},
    )
    return response.message.content


def cited_ids(draft: str) -> set[int]:
    """Extract cited answer_ids from a draft via the [id] bracket pattern —
    used by the Phase 4 faithfulness check to verify every citation the model
    emitted actually appears in the retrieved set (rule 1 says never invent an
    id, but LLMs occasionally do anyway, so this needs to be checked, not
    assumed)."""
    return {int(m) for m in CITATION_RE.findall(draft)}


def cited_sub_ids(draft: str) -> set[str]:
    """Sub-id counterpart of cited_ids(), for the chunk-concatenation
    citation scheme (CONCAT_SYSTEM_PROMPT) — matches "<digits><letter>"
    instead of a bare integer."""
    return set(CITATION_SUBID_RE.findall(draft))


def main() -> None:
    query = " ".join(sys.argv[1:]) or "How do I reset my password?"
    chunks = retrieve(query)
    draft = generate(query, chunks)
    print(draft)
    print("\n--- cited ids:", sorted(cited_ids(draft)), "---")
    get_client().close()


if __name__ == "__main__":
    main()
