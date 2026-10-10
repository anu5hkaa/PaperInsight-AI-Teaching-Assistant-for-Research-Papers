# Online eval: checks REAL student requests recorded in LangSmith (no ground truth).
#   1. an LLM judge scores new Q&A answers (faithful, no_contradiction, clear) and saves it as feedback
#   2. prints a report: requests, helpful rate, faithfulness, abstain rate, latency, errors
#   3. writes bad cases to results/online_review.json -> review them and add them to test_set.json
# Usage: python src/pdf_processing/evaluation/online_eval.py 7      (last 7 days)
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "retrieval"))

from langsmith import Client
from evaluate import judge_answer, NOT_FOUND_TEXT, average, percentile   # also turns tracing off for the judge

PROJECT = os.getenv("LANGSMITH_PROJECT", "default")
FEATURES = ["answer_question", "summarize_paper", "compare_papers"]
ONLINE_KEYS = ["faithful", "no_contradiction", "clear"]   # checks that need no correct answer
REVIEW_FILE = EVAL_DIR / "results" / "online_review.json"

client = Client()


def judge_run(run, feedback):
    # score one real Q&A answer with the judge, once
    answer = (run.outputs or {}).get("answer") or ""
    if run.name != "answer_question" or not answer or "faithful" in feedback:
        return

    context = "\n\n".join(s["document"] for s in run.outputs.get("sources", []))
    scores = judge_answer(run.inputs.get("question"), "(no reference answer for real questions)", answer, context) or {}

    for key in ONLINE_KEYS:
        if scores.get(key) is not None:
            client.create_feedback(run.id, key=key, score=scores[key])
            feedback[key] = scores[key]


def main(days):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    runs = [r for r in client.list_runs(project_name=PROJECT, is_root=True, start_time=since) if r.name in FEATURES]
    print(f"{len(runs)} student requests in the last {days} days (project: {PROJECT})\n")
    if not runs:
        return

    rows = []
    for run in runs:
        feedback = {f.key: f.score for f in client.list_feedback(run_ids=[run.id])}
        judge_run(run, feedback)

        answer = (run.outputs or {}).get("answer") or ""
        rows.append({
            "run_id": str(run.id),
            "feature": run.name,
            "inputs": run.inputs,
            "answer": answer,
            "latency": (run.end_time - run.start_time).total_seconds() if run.end_time else None,
            "error": run.error,
            "user_score": feedback.get("user_score"),
            "abstained": NOT_FOUND_TEXT in answer.lower() if run.name == "answer_question" else None,
            **{key: feedback.get(key) for key in ONLINE_KEYS},
        })

    latencies = [r["latency"] for r in rows if r["latency"] is not None]
    report = {
        "requests": len(rows),
        "helpful_rate": average([r["user_score"] for r in rows]),
        "ratings_given": sum(r["user_score"] is not None for r in rows),
        "faithfulness": average([r["faithful"] for r in rows]),
        "no_contradiction": average([r["no_contradiction"] for r in rows]),
        "clarity": average([r["clear"] for r in rows]),
        "abstain_rate": average([r["abstained"] for r in rows]),
        "error_rate": average([r["error"] is not None for r in rows]),
        "latency_p50": percentile(latencies, 50) if latencies else None,
        "latency_p95": percentile(latencies, 95) if latencies else None,
    }

    print("========== ONLINE REPORT ==========")
    for key, value in report.items():
        print(f"{key:18} {round(value, 3) if isinstance(value, float) else value}")

    # bad cases for a human to review; after review they become new offline test questions
    review = [
        {**r, "reason": reason, "correct_answer": "", "evidence": []}
        for r in rows
        for reason in [
            "not helpful" if r["user_score"] == 0 else
            "unsupported claim" if r["faithful"] == 0 else
            "contradicts paper" if r["no_contradiction"] == 0 else
            "error" if r["error"] else None
        ]
        if reason
    ]

    REVIEW_FILE.parent.mkdir(exist_ok=True)
    REVIEW_FILE.write_text(json.dumps(review, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(review)} cases to review -> {REVIEW_FILE}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
