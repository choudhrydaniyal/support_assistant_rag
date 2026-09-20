"""
Runs the eval set against the live pipeline (retrieval + escalation logic,
and generation if GOOGLE_API_KEY is set) and reports:

  - Retrieval hit-rate: did any expected chunk id appear in top-k for
    questions that should be answerable from the corpus?
  - Escalation-trigger accuracy: for questions with a known expected
    escalation outcome (True/False), did the system escalate correctly?
    Questions with expect_escalate=null are judgment calls (stale-info /
    ambiguous) and are reported but not scored pass/fail -- see README eval
    plan for why these need human or LLM-judge review instead of a hard rule.

This is a small, fast, deterministic layer of the eval plan. It does NOT
measure hallucination or answer correctness against the free-text generated
answer -- that requires either human review or an LLM-as-judge pass, both
described in README.md.

Usage:
    python eval/run_eval.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from assistant import SupportAssistant  # noqa: E402

EVAL_PATH = Path(__file__).parent / "eval_questions.jsonl"


def load_eval_set():
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    cases = load_eval_set()
    assistant = SupportAssistant()

    scored = 0
    correct = 0
    retrieval_scored = 0
    retrieval_hit = 0

    print(f"{'ID':5} {'CATEGORY':22} {'ESCALATED':10} {'EXPECTED':10} {'RETRIEVAL_HIT':14} RESULT")
    print("-" * 90)

    for case in cases:
        turn = assistant.ask(case["question"])

        retrieval_result = "n/a"
        expected_ids = case.get("expect_any_of_ids") or []
        if expected_ids:
            retrieval_scored += 1
            hit = any(cid in turn.retrieved_ids for cid in expected_ids)
            retrieval_hit += hit
            retrieval_result = "HIT" if hit else "MISS"

        expected_escalate = case.get("expect_escalate")
        result = "—"
        if expected_escalate is not None:
            scored += 1
            ok = turn.escalate == expected_escalate
            correct += ok
            result = "PASS" if ok else "FAIL"

        print(
            f"{case['id']:5} {case['category']:22} {str(turn.escalate):10} "
            f"{str(expected_escalate):10} {retrieval_result:14} {result}"
        )

    print("-" * 90)
    if scored:
        print(f"Escalation-trigger accuracy: {correct}/{scored} ({100*correct/scored:.0f}%)")
    if retrieval_scored:
        print(f"Retrieval hit-rate (expected id in top-{4}): "
              f"{retrieval_hit}/{retrieval_scored} ({100*retrieval_hit/retrieval_scored:.0f}%)")


if __name__ == "__main__":
    main()
