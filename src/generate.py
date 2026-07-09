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

ABSTENTION_MESSAGE = """\
Dear customer,

Thank you for reaching out. Unfortunately, we don't have information on \
that in our records. Please contact our support team directly so they can \
look into this for you.

Best regards,
Customer Support"""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    return "\n".join(f"[{chunk.answer_id}] {chunk.answer}" for chunk in chunks)


def build_user_message(query: str, chunks: list[RetrievedChunk]) -> str:
    return (
        f'Customer email:\n"""\n{query}\n"""\n\n'
        f"Context:\n{build_context_block(chunks)}"
    )


def generate(query: str, chunks: list[RetrievedChunk]) -> str:
    """Low temperature (0.2) since this is grounded generation, not creative
    writing — we want the model to consistently follow the citation/abstention
    rules rather than vary its phrasing run to run."""
    if not chunks or chunks[0].score < ABSTAIN_SCORE_THRESHOLD:
        return ABSTENTION_MESSAGE

    response = ollama.chat(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(query, chunks)},
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


def main() -> None:
    query = " ".join(sys.argv[1:]) or "How do I reset my password?"
    chunks = retrieve(query)
    draft = generate(query, chunks)
    print(draft)
    print("\n--- cited ids:", sorted(cited_ids(draft)), "---")
    get_client().close()


if __name__ == "__main__":
    main()
