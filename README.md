# LearnForge Support Assistant (prototype)

A retrieval-augmented customer support assistant for LearnForge, an ed-tech
company. Answers questions from a knowledge base of FAQs, policies, and past
support tickets; escalates to a human when it isn't confident.

Built for a take-home assignment. See `docs/architecture.md` for the system
diagram and `docs/schema.md` for the data schema.

## Quickstart

```bash
pip install -r requirements.txt
python src/ingest.py            # parses data/raw/*.md -> data/chunks.jsonl
export GOOGLE_API_KEY=...       # optional -- free key at https://aistudio.google.com/apikey
python src/cli.py               # interactive chat
python eval/run_eval.py         # automated eval
```

Without `GOOGLE_API_KEY` set, the assistant runs on a template-based
`MockLLM` so the retrieval and escalation logic can be exercised with zero
setup — useful for a first look, not a substitute for testing the real
model. All eval numbers below were run with the mock (retrieval/escalation
layer only); generation quality against the real model has not been
separately scored in this submission window.

## What it does

1. Retrieves the most relevant knowledge-base chunks for the user's message
   (TF-IDF cosine similarity, in-process, no external calls).
2. If nothing clears a similarity threshold, escalates immediately —
   the LLM is never called, so it can't hallucinate an answer to something
   not in the knowledge base.
3. Otherwise, sends the retrieved chunks + a system prompt (own words only,
   don't restate superseded claims as current, surface conflicts, admit
   uncertainty) to Gemini and returns the answer with citations.
4. Flags the turn for escalation if retrieval was weak, sources conflicted,
   the model expressed uncertainty, or the user asked for a human directly.

## Failure handling

| Failure mode | How it's handled |
|---|---|
| **Low-confidence retrieval** (question outside the knowledge base, e.g. "what's the weather") | Similarity threshold gate in `assistant.py`. Below threshold, the system escalates without calling the LLM at all — cheaper and removes the main hallucination surface. |
| **Stale / superseded data** (the sample corpus deliberately includes self-correcting text, e.g. "an older version said X; that is outdated, current policy is Y") | Ingestion flags any chunk containing a staleness marker (`contains_superseded_claim`). The system prompt explicitly instructs the model to follow the correction, not the original claim, and never present a superseded fact as current. |
| **Genuinely conflicting sources** (e.g. TICKET-08: a user is told 14 days by one channel and "no subscription refunds after renewal" by another, and neither is marked as the fix) | A lightweight heuristic (`detect_conflicting_sources`) scans top-k chunks for differing concrete values (duration counts) not already flagged as superseded, and flags `CONFLICTING_SOURCES` so the turn escalates instead of picking one answer with false confidence. |
| **Bad / ambiguous retrieval** (vague query, e.g. ticket-07-style "cancel my LearnForge") | Not solved by retrieval alone. The system prompt asks the model to ask a clarifying question rather than guess when the request is underspecified; the eval set includes this case (E13) as a category the automated harness reports on but does not hard-score, because "did it ask a good clarifying question" needs human/LLM-judge review, not a keyword check. |
| **Model expresses its own uncertainty** | Post-hoc regex check on the generated answer for uncertainty phrases (`don't have enough information`, `not sure`, etc.) triggers `MODEL_UNCERTAIN` escalation, as a backstop for cases the retrieval-confidence gate didn't catch. |
| **User explicitly asks for a person** | Pattern-matched before retrieval even runs; short-circuits straight to escalation. |

Escalation in this prototype means: the API call returns
`escalate=True` plus a machine-readable list of reasons
(`turn.escalation_reasons`). Wiring that to an actual human queue (Zendesk,
Intercom, etc.) is a downstream integration not built here — this
prototype's job is producing the right signal, not routing it.

## Eval plan

Answer quality on a RAG support bot has two genuinely different failure
axes, and they need different measurement:

**1. Retrieval quality — automated, in this repo (`eval/run_eval.py`)**
A fixed 15-question set (`eval/eval_questions.jsonl`) covering answerable
questions, conflicting-policy questions, stale-info questions, out-of-scope
questions, and one ambiguous query, each with an expected top-k chunk id
(where applicable) and expected escalation outcome. The harness reports:
- **Retrieval hit-rate**: did an expected chunk id appear in top-4.
- **Escalation-trigger accuracy**: did `escalate` match the expected
  True/False for the cases where the right answer is unambiguous (e.g.
  out-of-scope questions must escalate; clean answerable questions must
  not). Current run: 11/12 scored cases correct, 91% retrieval hit-rate —
  see "known limitation" below for the one failure.

This layer is cheap, deterministic, and safe to run on every change (a
regression test, effectively), but it does not tell you if the generated
*text* is actually correct or grounded.

**2. Answer correctness / hallucination rate — needs LLM-as-judge or human
review, not yet built here.** The plan for a real deployment:
- Maintain a growing set of (question, gold answer, gold citations) pairs,
  ideally sourced from real support transcripts.
- For each pipeline answer, run a judge prompt (separate model call) that
  checks: (a) every factual claim is supported by the cited chunk text
  (groundedness), (b) the answer doesn't contradict a more recent/correcting
  chunk when one exists, (c) semantic match against the gold answer.
- Track hallucination rate as: fraction of answers containing a claim not
  traceable to a retrieved chunk. This needs the judge to have access to the
  same retrieved context the generator had, not just the raw corpus, or it
  will conflate "wrong chunk was retrieved" with "model invented something
  from a chunk it did have."
- Track escalation precision/recall against human-labeled "should have
  escalated" ground truth once real usage data exists — the current
  threshold (0.12) and heuristics are tuned by inspection on 15 synthetic
  questions, not a labeled set, and should be expected to need recalibration
  against real traffic.
- Spot-check with human review on a sample each week regardless of
  automated scores; support-domain correctness (e.g. "is this actually
  LearnForge's refund policy") is a business fact, not a text-similarity
  fact, and no automated judge should be the last check on it.

**Known limitation surfaced by the current eval run:** question E01 ("how
many days do I have to request a refund") triggers `CONFLICTING_SOURCES`
even though it's meant to be a clean answerable case — the retrieved set
includes both the 14-day standard policy and TICKET-03's mention of a
30-day promotional guarantee, and the heuristic can't yet tell "different
promo terms can coexist" from "these actually contradict." This is the
right failure direction (over-flagging a genuine ambiguity rather than
silently picking 14 days) but it's a precision problem worth fixing with a
better heuristic or a small classifier before this ships.

## Trade-offs (what I'd change with more time/budget, and why)

- **TF-IDF instead of dense embeddings.** Chosen because the sample corpus
  is 40 short, keyword-rich chunks, where TF-IDF is competitive with
  embeddings and adds zero API cost/latency/network dependency to
  retrieval. It will not scale semantically — it can't match "I can't log
  in" to a chunk that only says "authentication failure," where an
  embedding model would. With more time: swap `src/retriever.py`'s
  `TfidfVectorizer` for Gemini's `text-embedding-004` behind the same
  `.search()` interface, backed by a real vector DB (pgvector is the
  pragmatic choice at this scale; Pinecone/Weaviate if multi-tenant scale is
  needed later) — the rest of the pipeline doesn't need to change.
- **Heuristic escalation gates instead of a learned confidence model.**
  Chosen for interpretability and because there's no labeled escalation
  data to train on yet — every threshold here is inspectable and
  justifiable in a design review. Cost: thresholds are hand-tuned on 15
  synthetic examples and will drift as real traffic differs from that
  sample (see eval plan). With more time: log real escalation
  decisions + outcomes, and either recalibrate thresholds or train a small
  classifier once there's enough labeled data.
- **No query rewriting for multi-turn context.** Conversation history is
  stored (`SupportAssistant.history`) and passed to the LLM for generation,
  but retrieval runs on the latest raw user message only. A user saying
  "what about on iOS?" after asking about offline downloads will retrieve
  poorly because "what about on iOS" alone is a weak query. Fix: an
  LLM call (or even a cheap heuristic) that rewrites the query using the
  last turn before retrieval — deferred because it's an extra LLM call
  (cost/latency) that needed to be weighed against the 4-6 hour budget, and
  the sample data doesn't have enough genuinely multi-turn conversational
  cases to validate it well.
- **Keyword-based staleness/conflict detection instead of explicit metadata
  authored at the source.** Works here because the sample data conveniently
  self-annotates ("that wording is outdated..."). A real knowledge base
  won't reliably do that. Real fix: require content owners to mark
  `superseded_by` relationships explicitly (see `docs/schema.md`) when they
  update a policy, and treat undated/unmarked contradictions as a content
  gap to flag back to the knowledge-base owners, not just something the
  runtime system papers over.
- **One entry = one chunk, no overlap/windowing.** Fine for this corpus
  where entries are already short and self-contained. Won't hold for long
  source documents (a 20-page course policy PDF); that needs section-aware
  chunking with overlap, which I did not need to build for this data but
  called out in `docs/schema.md` as the extension point.
- **TICKET chunks are included in the same retrievable corpus as customer-facing
  FAQs/policies.** This was a conscious simplification, not a default I'd
  ship: past tickets contain other users' text and (given the metadata
  design) could be excluded from a customer-facing bot via a `doc_type`
  filter while remaining available to an agent-facing "similar past ticket"
  tool. I left both in one corpus in the prototype to keep the retrieval
  demo simple to inspect end-to-end within the time budget, and note the
  data-schema field (`doc_type`) that makes the split trivial to add.
- **MockLLM fallback.** Added so the reviewer can run the whole pipeline
  without provisioning a key first, and so retrieval/escalation logic is
  independently testable from generation quality. Real generation quality
  (tone, correctness, citation accuracy) has not been evaluated against the
  live Gemini API in this submission — that's the highest-priority next
  step before trusting this beyond a demo.

## Repo layout

```
data/raw/            source markdown (faqs.md, policies.md, tickets.md)
data/chunks.jsonl    generated by src/ingest.py
src/ingest.py        markdown -> structured chunks
src/retriever.py     TF-IDF retrieval
src/llm.py           Gemini client + MockLLM fallback
src/assistant.py     RAG loop + escalation state machine
src/cli.py           interactive multi-turn chat
eval/eval_questions.jsonl
eval/run_eval.py     automated retrieval + escalation scoring
docs/architecture.md system design diagram
docs/schema.md       chunk / vector-DB record schema
```
