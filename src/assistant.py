"""
Core retrieval + generation loop, with escalation gating.

Escalation triggers (any one is sufficient):
  1. LOW_RETRIEVAL_CONFIDENCE — best chunk similarity below RETRIEVAL_THRESHOLD.
     The model is never called in this case: if retrieval can't find anything
     relevant, generation can only hallucinate or restate the question.
  2. CONFLICTING_SOURCES — top retrieved chunks disagree on a concrete fact
     (heuristic: distinct numeric/duration tokens across chunks not flagged
     as superseded, e.g. "14 days" vs "30 days"). Surfaced to the model via
     the system prompt; also flagged to the caller.
  3. MODEL_UNCERTAIN — the model's own answer contains an uncertainty phrase
     ("don't have enough information", "not sure", "escalate", etc).
  4. EXPLICIT_REQUEST — user asks for a human directly.

This is intentionally a small set of cheap, inspectable heuristics rather
than a learned confidence model -- see README trade-offs for why.
"""
import re
from dataclasses import dataclass, field

from retriever import Retriever
from llm import get_llm

RETRIEVAL_THRESHOLD = 0.12
TOP_K = 4

UNCERTAINTY_PATTERNS = [
    r"don't have enough information",
    r"not confident",
    r"not sure",
    r"connect you (with|to) a human",
    r"human agent",
    r"escalate",
]

HUMAN_REQUEST_PATTERNS = [
    r"\btalk to a (human|person|agent)\b",
    r"\breal person\b",
    r"\bspeak (to|with) someone\b",
]

DURATION_RE = re.compile(r"\b(\d+)[\s-]*day\b", re.IGNORECASE)


def detect_conflicting_sources(chunks):
    """Heuristic: collect day-count claims from chunks NOT already flagged as
    superseded; if more than one distinct value appears, treat as a live
    conflict worth flagging (rather than a self-corrected document)."""
    values = set()
    for c in chunks:
        if c.get("contains_superseded_claim"):
            continue
        for m in DURATION_RE.finditer(c["text"]):
            values.add(int(m.group(1)))
    return len(values) > 1


@dataclass
class Turn:
    user_message: str
    answer: str
    retrieved_ids: list = field(default_factory=list)
    top_score: float = 0.0
    escalate: bool = False
    escalation_reasons: list = field(default_factory=list)


class SupportAssistant:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = get_llm()
        self.history: list[Turn] = []

    def ask(self, user_message: str) -> Turn:
        reasons = []

        if any(re.search(p, user_message, re.IGNORECASE) for p in HUMAN_REQUEST_PATTERNS):
            turn = Turn(
                user_message=user_message,
                answer="Of course -- connecting you with a human support agent now.",
                escalate=True,
                escalation_reasons=["EXPLICIT_REQUEST"],
            )
            self.history.append(turn)
            return turn

        hits = self.retriever.search(user_message, top_k=TOP_K)
        top_score = hits[0]["score"] if hits else 0.0

        if top_score < RETRIEVAL_THRESHOLD:
            turn = Turn(
                user_message=user_message,
                answer=(
                    "I couldn't find anything in the knowledge base that "
                    "confidently answers that. I'll connect you with a human "
                    "agent who can help further."
                ),
                retrieved_ids=[h["id"] for h in hits],
                top_score=top_score,
                escalate=True,
                escalation_reasons=["LOW_RETRIEVAL_CONFIDENCE"],
            )
            self.history.append(turn)
            return turn

        if detect_conflicting_sources(hits):
            reasons.append("CONFLICTING_SOURCES")

        answer = self.llm.generate(hits, user_message)

        if any(re.search(p, answer, re.IGNORECASE) for p in UNCERTAINTY_PATTERNS):
            reasons.append("MODEL_UNCERTAIN")

        turn = Turn(
            user_message=user_message,
            answer=answer,
            retrieved_ids=[h["id"] for h in hits],
            top_score=top_score,
            escalate=len(reasons) > 0,
            escalation_reasons=reasons,
        )
        self.history.append(turn)
        return turn
