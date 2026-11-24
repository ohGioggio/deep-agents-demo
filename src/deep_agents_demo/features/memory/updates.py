"""Deciding how a new fact changes what is already remembered: add, update, delete or nothing."""
import re
import unicodedata
import warnings
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

RECENT_MESSAGES = 6
UPDATE_CANDIDATES = 5

DECISION_PROMPT = """You keep a long-term memory about one user. A new fact is about to be stored.
Compare it with the existing memories (candidates) and the recent conversation, then answer with one operation:

- ADD: the fact is new, or is about something unrelated to every candidate.
  Example: the memory says "Likes green tea" and the fact is "Has a dog named Rex" -> ADD.
- UPDATE: the fact changes a candidate, so the candidate is now out of date.
  Give memory_id and the full new text of the memory, not only the changed part.
  Example: the memory "[3] Lives in Rome" and the fact "Now lives in Milan" -> UPDATE, memory_id "3",
  text "Lives in Milan".
- NONE: a candidate already says the same thing, perhaps in other words.
  Example: the memory "[5] Is vegetarian" and the fact "Doesn't eat meat" -> NONE.
  Set memory_id to the id of the memory that already says it.
- DELETE: the user retracts a candidate, so it must no longer be remembered.
  Example: the memory "[2] Works at Acme" and the user says "I never worked at Acme" -> DELETE, memory_id "2".

Rules:
- Use memory_id only with an id from the candidate list. Never invent an id.
- A similar but different fact is ADD, not UPDATE. "Likes cats" and "Likes dogs" are two facts.
- Write text as a short third-person fact, such as "Lives in Milan".
- Give a short reason.
"""


class Operation(BaseModel):
    """What to do with a new fact, as decided by the model."""

    kind: Literal["ADD", "UPDATE", "DELETE", "NONE"]
    memory_id: str | None = None
    text: str | None = None
    reason: str = ""


def _words(text: str) -> list[str]:
    """Lowercase word tokens, so case, punctuation and how an accent is encoded (NFC or NFD)
    do not block a match."""
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).lower())


def _bm25_retriever():
    """BM25Retriever, imported on first use: langchain_community is slow to import and noisy."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`langchain-community` is being sunset",
            category=DeprecationWarning,
        )
        from langchain_community.retrievers import BM25Retriever
    return BM25Retriever


def rank_candidates(fact: str, memories: list[dict], k: int = UPDATE_CANDIDATES) -> list[dict]:
    """The at most ``k`` memories most similar to ``fact``, best first (BM25 over the memory texts)."""
    if not memories:
        return []
    retriever = _bm25_retriever().from_texts(
        [memory["memory"] for memory in memories],
        metadatas=[{"id": memory["id"]} for memory in memories],
        preprocess_func=_words,
        k=k,
    )
    return [{"id": doc.metadata["id"], "memory": doc.page_content} for doc in retriever.invoke(fact)]


class MemoryDecider:
    """One structured-output call to a chat model per new fact."""

    def __init__(self, model: BaseChatModel):
        self.model = model

    def decide(self, fact: str, recent: list[str], candidates: list[dict]) -> Operation:
        """Return the operation. Errors from the model propagate; the caller falls back to ADD."""
        listed = "\n".join(f"[{candidate['id']}] {candidate['memory']}" for candidate in candidates)
        conversation = "\n".join(f"- {message}" for message in recent) or "(none)"
        user = (
            f"Recent messages:\n{conversation}\n\n"
            f"Candidates:\n{listed or '(none)'}\n\n"
            f"New fact: {fact}"
        )
        messages = [SystemMessage(content=DECISION_PROMPT), HumanMessage(content=user)]
        return self.model.with_structured_output(Operation).invoke(messages)
