"""
Ingest raw knowledge-base markdown files into a flat chunk store.

Design choice: one FAQ / policy / ticket entry = one chunk. Each entry is
already a coherent, self-contained unit (a single question, a single policy
section, a single ticket transcript). Splitting further would break the
context a model needs to answer correctly; merging entries together would
dilute retrieval precision. See docs/schema.md for the full rationale.

Usage:
    python src/ingest.py
Writes: data/chunks.jsonl
"""
import json
import re
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_PATH = Path(__file__).parent.parent / "data" / "chunks.jsonl"

# Only these header patterns are real entries. The source files contain a
# couple of stray "SECTION N — ..." header lines left over from how the
# original email attachment was split into three files; those are noise
# and must not become chunks.
HEADER_RE = re.compile(r"^#\s*(FAQ|POLICY|TICKET)-(\d+)\s*—\s*(.+)$")

DATE_RE = re.compile(
    r"(Last reviewed|Last updated|Effective date|Effective|Updated|Reviewed)\s*:\s*([A-Za-z]+ \d{4}|\d{4})",
    re.IGNORECASE,
)

STALENESS_MARKERS = [
    "no longer", "outdated", "obsolete", "older version", "older article",
    "older documentation", "previous version", "previous mobile help article",
    "archived documentation", "retired", "should not be treated as",
    "is outdated", "that wording is outdated", "this is no longer",
    "that information is obsolete", "that plan is no longer offered",
]

STATUS_RE = re.compile(r"^STATUS:\s*(.+)$", re.MULTILINE)

DOC_TYPE_MAP = {"FAQ": "faq", "POLICY": "policy", "TICKET": "ticket"}


def split_entries(raw_text: str):
    """Split a raw file into (header_line, body_text) blocks on level-1 headers."""
    lines = raw_text.splitlines()
    blocks = []
    current_header = None
    current_body = []
    for line in lines:
        if line.startswith("# "):
            if current_header is not None:
                blocks.append((current_header, "\n".join(current_body).strip()))
            current_header = line.strip()
            current_body = []
        else:
            if line.strip() == "---":
                continue
            current_body.append(line)
    if current_header is not None:
        blocks.append((current_header, "\n".join(current_body).strip()))
    return blocks


def extract_metadata(doc_type: str, body: str):
    meta = {}

    date_match = DATE_RE.search(body)
    if date_match:
        meta["last_updated_raw"] = date_match.group(2)
    else:
        meta["last_updated_raw"] = None

    meta["contains_superseded_claim"] = any(
        marker.lower() in body.lower() for marker in STALENESS_MARKERS
    )

    if doc_type == "ticket":
        status_match = STATUS_RE.search(body)
        meta["ticket_status"] = status_match.group(1).strip() if status_match else None

    return meta


def ingest_file(path: Path, source_label: str):
    raw = path.read_text(encoding="utf-8")
    chunks = []
    for header, body in split_entries(raw):
        m = HEADER_RE.match(header)
        if not m:
            # Stray / malformed header (e.g. leftover "SECTION 2 ..." marker).
            # Not a real entry -> skip.
            continue
        type_prefix, number, title = m.groups()
        doc_type = DOC_TYPE_MAP[type_prefix]
        chunk_id = f"{type_prefix}-{number.zfill(2)}"
        metadata = extract_metadata(doc_type, body)
        chunks.append({
            "id": chunk_id,
            "doc_type": doc_type,
            "title": title.strip(),
            "text": body.strip(),
            "source_file": source_label,
            **metadata,
        })
    return chunks


def main():
    all_chunks = []
    all_chunks += ingest_file(RAW_DIR / "faqs.md", "faqs.md")
    all_chunks += ingest_file(RAW_DIR / "policies.md", "policies.md")
    all_chunks += ingest_file(RAW_DIR / "tickets.md", "tickets.md")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_chunks)} chunks to {OUT_PATH}")
    by_type = {}
    for c in all_chunks:
        by_type[c["doc_type"]] = by_type.get(c["doc_type"], 0) + 1
    print("By type:", by_type)
    flagged = sum(1 for c in all_chunks if c["contains_superseded_claim"])
    print(f"Chunks flagged as containing a superseded/outdated claim: {flagged}")


if __name__ == "__main__":
    main()
