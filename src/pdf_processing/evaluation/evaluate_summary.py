# Offline eval for paper summaries: key-point coverage, faithfulness, clarity, latency.
# Usage: python src/pdf_processing/evaluation/evaluate_summary.py summary_v1
import os
import re
import sys
import json
import time
from pathlib import Path

os.environ["LANGSMITH_TRACING"] = "false"   # keep offline eval runs out of the online project

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "retrieval"))

from summarizer import summarize_paper
from retriever import warm_up
from generator import generate_answer
from config import JUDGE_MODEL

TEST_SET_FILE = EVAL_DIR / "summary_test_set.json"
RESULTS_DIR = EVAL_DIR / "results"
TEACHER_HEADER = re.compile(r"teacher.{0,2}s takeaway", re.IGNORECASE)


def judge_summary(paper, key_points, summary, context):
    # only the part before the teacher's takeaway must be faithful to the paper
    match = TEACHER_HEADER.search(summary)
    grounded = summary[:match.start()] if match else summary
    points = "\n".join(f"{i}. {p}" for i, p in enumerate(key_points, start=1))

    prompt = f"""
You are grading a summary of the research paper "{paper}".

Key points a good summary should contain:
{points}

Context the AI was given:
{context}

Summary (grounded part):
{grounded}

Return:
- covered: list of the numbers of the key points that the summary clearly contains
- faithful: 1 if every factual claim in the summary is supported by the context, else 0
- clear: 1 if a beginner student could follow the summary, else 0

Return ONLY JSON like this: {{"covered": [1, 2, 4], "faithful": 1, "clear": 1}}
"""
    raw = generate_answer(prompt, model=JUDGE_MODEL)
    if raw is None:
        return {}
    try:
        return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except json.JSONDecodeError:
        print("Judge returned bad JSON:", raw)
        return {}


def main(run_name):
    warm_up()
    test_set = json.loads(TEST_SET_FILE.read_text(encoding="utf-8"))

    rows = []
    for item in test_set:
        start = time.time()
        result = summarize_paper(item["paper"])
        total_time = time.time() - start

        context = "\n\n".join(s["document"] for s in result["sources"])
        scores = judge_summary(item["paper"], item["key_points"], result["summary"], context)
        covered = [n for n in scores.get("covered", []) if 1 <= n <= len(item["key_points"])]

        row = {
            "paper": item["paper"],
            "summary": result["summary"],
            "coverage": len(set(covered)) / len(item["key_points"]),
            "missing": [p for i, p in enumerate(item["key_points"], start=1) if i not in covered],
            "faithful": scores.get("faithful"),
            "clear": scores.get("clear"),
            "total_time": total_time,
            "timings": result["timings"],
        }
        print(f"{item['paper']:28} coverage={row['coverage']:.2f} faithful={row['faithful']} "
              f"clear={row['clear']} {total_time:.1f}s")
        for point in row["missing"]:
            print(f"    missing: {point}")
        rows.append(row)

    def average(key):
        values = [r[key] for r in rows if r[key] is not None]
        return sum(values) / len(values) if values else None

    summary = {
        "run": run_name,
        "coverage": average("coverage"),
        "faithfulness": average("faithful"),
        "clarity": average("clear"),
        "latency_avg": average("total_time"),
        "latency_max": max(r["total_time"] for r in rows),
    }

    print("\n========== SUMMARY ==========")
    for key, value in summary.items():
        print(f"{key:14} {round(value, 3) if isinstance(value, float) else value}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{run_name}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to: {RESULTS_DIR / run_name}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "summary")
