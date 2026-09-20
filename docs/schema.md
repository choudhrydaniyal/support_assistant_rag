# Data Schema

## Chunking decision

One knowledge-base entry (one FAQ, one policy section, one ticket
transcript) = one chunk. Each entry in the sample data is already a
coherent, self-contained unit of 100–400 words. Sub-splitting would cut a
policy's caveat away from its main claim (e.g. separating "14-day refund
window" from "unless a technical problem prevented access"); merging
entries would dilute TF-IDF/embedding similarity and blur citations. For a
larger real corpus (e.g. a 40-page course catalog PDF) the same principle
still applies, just at a different chunk size: split on natural section
boundaries, not fixed token windows, and keep a caveat with its claim.

## Chunk record (current implementation: `data/chunks.jsonl`)

```json
{
  "id": "POLICY-02",
  "doc_type": "policy",
  "title": "Cancellation and Refund Policy",
  "text": "You can cancel an active subscription at any time ...",
  "source_file": "policies.md",
  "last_updated_raw": "January 2026",
  "contains_superseded_claim": true,
  "ticket_status": null
}
```

| Field | Type | Notes |
|---|---|---|
| `id` | string | Stable human-readable id, doubles as the citation shown to users. |
| `doc_type` | enum | `faq` \| `policy` \| `ticket`. Drives prompt framing and future access-control rules (e.g. tickets may contain PII and could be excluded from a customer-facing bot even though they're useful for support-agent-facing tooling). |
| `title` | string | Used in retrieval (concatenated with `text`) and shown in citations. |
| `text` | string | Full chunk body, source of truth for generation. |
| `source_file` | string | Provenance, for debugging and for a future "view source" link. |
| `last_updated_raw` | string \| null | Extracted from the "Last reviewed / Effective / Updated" line if present. Currently descriptive only; see trade-offs for what a freshness-ranking upgrade would do with it. |
| `contains_superseded_claim` | bool | Heuristic flag (keyword match on "outdated", "no longer", "older version", etc). Used by the conflict-detection heuristic and passed to the model's system prompt implicitly via the text itself. |
| `ticket_status` | string \| null | Ticket-only. `Resolved` / `Pending` / `Escalated` extracted from the transcript's `STATUS:` line. |

## Vector DB record (production target)

The prototype uses TF-IDF (see `src/retriever.py` docstring for why), but
the record shape is designed to map directly onto a real vector DB
(pgvector, Pinecone, Weaviate, etc.) with no schema change to the chunk
itself — only an `embedding` field is added and the corpus becomes a
proper index rather than an in-memory list:

```json
{
  "id": "POLICY-02",
  "embedding": [0.0123, -0.0456, ...],
  "metadata": {
    "doc_type": "policy",
    "title": "Cancellation and Refund Policy",
    "source_file": "policies.md",
    "last_updated": "2026-01-01",
    "contains_superseded_claim": true,
    "ticket_status": null
  },
  "text": "You can cancel an active subscription at any time ..."
}
```

Notes for the production version:
- `last_updated` becomes a real ISO date (parsed, not raw string) so it can
  be used for **recency-boosted ranking**, not just display.
- Metadata filtering (`doc_type != "ticket"` for the customer-facing bot,
  `doc_type == "ticket" AND ticket_status == "Resolved"` for a support-agent
  tool that wants precedent) becomes a first-class query filter rather than
  a Python-side check.
- A `superseded_by` field (pointer to the correcting chunk id) would let the
  conflict heuristic in `assistant.py` become exact instead of approximate —
  today it infers "self-corrected" from keyword phrases in the same chunk;
  in production, whoever authors/updates the knowledge base should be able
  to explicitly mark "this replaces POLICY-02-v1".
