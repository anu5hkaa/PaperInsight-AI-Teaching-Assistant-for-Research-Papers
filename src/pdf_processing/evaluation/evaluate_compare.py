# Offline eval for comparing 2-3 papers: retrieval per paper, correctness,
# attribution (no mixing up papers), faithfulness, latency.
# Usage: python src/pdf_processing/evaluation/evaluate_compare.py compare_v1
import os
import re
import sys
import json
import time
from pathlib import Path

os.environ["LANGSMITH_TRACING"] = "false"   # keep offline eval runs out of the online project

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "retrieval"))

from comparision import compare_papers
from generator import generate_answer
from retriever import warm_up
from config import JUDGE_MODEL

TEST_SET_FILE = EVAL_DIR / "compare_test_set.json"
RESULTS_DIR = EVAL_DIR / "results"
TEACHER_HEADER = re.compile(r"teacher.{0,2}s takeaway", re.IGNORECASE)


def clean(text):
    return " ".join(text.split()).lower()


def judge_comparison(question, expected, answer, context):

    # only the part before the takeaway must be faithful to the papers
    match = TEACHER_HEADER.search(answer)
    grounded = answer[:match.start()] if match else answer

    expected_text = "\n".join(f"- {paper}: {fact}" for paper, fact in expected.items())

    prompt = f"""
You are grading an AI that compares research papers.

Question:
{question}

Expected key fact for each paper:
{expected_text}

Context the AI was given (grouped by paper):
{context}

AI comparison (grounded part):
{grounded}

Give 0 or 1 for each check:
- correct: the comparison states the expected key fact for EVERY paper (extra detail is fine)
- attribution: every fact is given to the right paper (1 if nothing is mixed up between papers)
- faithful: every factual claim is supported by the context

Return ONLY JSON like this: {{"correct": 1, "attribution": 1, "faithful": 1}}
"""
    raw = generate_answer(prompt, model=JUDGE_MODEL)
    if raw is None:
        return {}

    raw = raw[raw.find("{"): raw.rfind("}") + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("Judge returned bad JSON:", raw)
        return {}


def average(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main(run_name):

    warm_up()

    with open(TEST_SET_FILE, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    rows = []

    for item in test_set:

        print(f"\n[{item['id']}/{len(test_set)}] {item['question']}")

        start = time.time()
        result = compare_papers(item["question"], item["papers"], full=item["full"])
        total_time = time.time() - start

        answer = result["comparison"] or ""

        # retrieval check per paper: evidence must be in THAT paper's chunks
        paper_hits = {}
        for paper, evidence_list in item["evidence"].items():
            docs = [clean(source["document"]) for source in result["sources"][paper]]
            paper_hits[paper] = any(ev in doc for ev in evidence_list for doc in docs)

        context = "\n\n".join(
            f"===== PAPER: {paper} =====\n" + "\n\n".join(s["document"] for s in sources)
            for paper, sources in result["sources"].items()
        )

        scores = judge_comparison(item["question"], item["expected"], answer, context) if answer else {}

        row = {
            "id": item["id"],
            "question": item["question"],
            "papers": item["papers"],
            "answer": answer,
            "total_time": total_time,
            "timings": result["timings"],
            "paper_hits": paper_hits,
            "all_papers_found": all(paper_hits.values()),
            "correct": scores.get("correct", 0),
            "attribution": scores.get("attribution", 0),
            "faithful": scores.get("faithful", 0),
        }

        print({k: row[k] for k in ["paper_hits", "correct", "attribution", "faithful", "total_time"]})
        rows.append(row)

    times = sorted(r["total_time"] for r in rows)

    summary = {
        "run": run_name,
        "retrieval_all_papers": average([r["all_papers_found"] for r in rows]),
        "correctness": average([r["correct"] for r in rows]),
        "attribution": average([r["attribution"] for r in rows]),
        "faithfulness": average([r["faithful"] for r in rows]),
        "latency_p50": times[len(times) // 2],
        "latency_max": times[-1],
        "avg_retrieve_time": average([r["timings"]["retrieve"] for r in rows]),
        "avg_llm_time": average([r["timings"]["llm"] for r in rows]),
    }

    print("\n========== SUMMARY ==========")
    for key, value in summary.items():
        print(f"{key:22} {round(value, 3) if isinstance(value, float) else value}")

    # failure store: any check that failed
    failures = [r for r in rows if not (r["all_papers_found"] and r["correct"] and r["attribution"] and r["faithful"])]

    print(f"\n========== FAILURES ({len(failures)}) ==========")
    for r in failures:
        failed = [k for k in ["all_papers_found", "correct", "attribution", "faithful"] if not r[k]]
        print(f"Q{r['id']:>2}  {', '.join(failed):35} {r['question']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{run_name}.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=4, ensure_ascii=False)
    with open(RESULTS_DIR / f"{run_name}_failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=4, ensure_ascii=False)

    print(f"\nSaved to: {RESULTS_DIR / (run_name + '.json')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "compare")
