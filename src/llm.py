"""
LLM wrapper. Real backend: Google Gemini free tier (gemini-1.5-flash /
gemini-2.0-flash, whichever the current free alias is).

If GOOGLE_API_KEY is not set, falls back to a MockLLM that does template
generation over the retrieved context. This exists so the full RAG loop
(retrieval -> prompt construction -> escalation logic) is runnable and
testable without a network call / API key -- useful for CI and for anyone
reviewing this without wanting to provision a key first. It is NOT a
substitute for testing against the real model before trusting outputs.
"""
import os


SYSTEM_PROMPT = """You are LearnForge's customer support assistant.

Rules you must follow:
1. Answer ONLY using the CONTEXT provided below. Do not use outside knowledge
   about LearnForge or about ed-tech platforms in general.
2. Some context documents explicitly mark parts of themselves as outdated,
   superseded, or retired (phrases like "no longer", "outdated", "older
   version", "should not be treated as current"). Never present a
   superseded claim as current policy. If a document corrects itself
   in-line, follow the correction, not the original claim.
3. If two context documents genuinely conflict and neither is marked as the
   correction, say so explicitly rather than picking one silently.
4. If the context does not contain enough information to answer confidently,
   say you don't have enough information and that you'll connect the user to
   a human agent. Do not guess or fill gaps with plausible-sounding details.
5. Keep answers short and direct. Cite which FAQ/POLICY/TICKET id(s) you
   drew from in parentheses at the end, e.g. (FAQ-02, POLICY-02).
"""


def build_user_turn(context_chunks, user_message):
    context_block = "\n\n".join(
        f"[{c['id']}] {c['title']}\n{c['text']}" for c in context_chunks
    )
    return f"CONTEXT:\n{context_block}\n\nUSER QUESTION:\n{user_message}"


class GeminiLLM:
    def __init__(self, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai  # imported lazily
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model, system_instruction=SYSTEM_PROMPT)
        self._chat = self.model.start_chat(history=[])

    def generate(self, context_chunks, user_message) -> str:
        turn = build_user_turn(context_chunks, user_message)
        response = self._chat.send_message(turn)
        return response.text


class MockLLM:
    """
    Deterministic, template-based stand-in for the real model. Lets the
    pipeline (retrieval -> escalation -> multi-turn state) be exercised and
    unit-tested without network access or an API key.
    """

    def __init__(self):
        self.history = []

    def generate(self, context_chunks, user_message) -> str:
        self.history.append(("user", user_message))
        if not context_chunks:
            answer = (
                "I don't have enough information in the knowledge base to "
                "answer that confidently. I'll connect you with a human agent."
            )
        else:
            top = context_chunks[0]
            ids = ", ".join(c["id"] for c in context_chunks)
            answer = (
                f"[MOCK LLM -- set GOOGLE_API_KEY for a real answer]\n"
                f"Based on {top['id']} ({top['title']}), here is the most "
                f"relevant guidance found in the knowledge base. "
                f"({ids})"
            )
        self.history.append(("assistant", answer))
        return answer


def get_llm():
    if os.environ.get("GOOGLE_API_KEY"):
        try:
            return GeminiLLM()
        except Exception as e:
            print(f"[warn] Falling back to MockLLM: {e}")
    return MockLLM()
