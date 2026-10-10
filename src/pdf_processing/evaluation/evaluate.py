import os
import re
import sys
import json
import time
from pathlib import Path

os.environ["LANGSMITH_TRACING"] = "false"   # keep offline eval runs out of the online project

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "retrieval"))

from qa import answer_question
from retriever import warm_up
from generator import generate_answer
from config import JUDGE_MODEL   # a bigger model grades the answers

TEST_SET_FILE = EVAL_DIR / "test_set.json"
RESULTS_DIR = EVAL_DIR / "results"
NOT_FOUND_TEXT = "does not contain enough information"
JUDGE_KEYS = ["correct", "faithful", "no_contradiction", "clear", "example", "terms"]
TEACHER_HEADER = re.compile(r"teacher.{0,2}s explanation", re.IGNORECASE)


def clean(text):
    return " ".join(text.split()).lower()


def find_rank(evidence_list, chunks):
    # 1-based position of the first chunk that contains an evidence phrase
    for rank, chunk in enumerate(chunks, start=1):
        if any(evidence in clean(chunk) for evidence in evidence_list):
            return rank
    return None


def split_answer(answer):
    # grounded part (Answer + From the paper) and teaching part (explanation + key terms)
    match = TEACHER_HEADER.search(answer)
    if match is None:
        return answer, ""
    return answer[:match.start()], answer[match.start():]


def cited_pages(answer):
    return {int(p) for p in re.findall(r"page\s*(\d+)", answer.lower())}


def judge_answer(question, reference, answer, context):
    grounded, teaching = split_answer(answer)
    prompt = f"""
You are grading an AI teacher that answers questions about a research paper.

Question:
{question}

Reference answer (or key points):
{reference}

Context the AI was given:
{context}

GROUNDED PART of the AI answer (must use only the context):
{grounded}

TEACHING PART of the AI answer (may use general knowledge):
{teaching or "(none)"}

Give 0 or 1 for each check:
- correct: the answer contains the key facts or most key points of the reference (extra detail is fine)
- faithful: every factual claim in the GROUNDED PART is supported by the context
- no_contradiction: nothing in the TEACHING PART contradicts the context (1 if there is no teaching part)
- clear: a beginner student could understand the answer
- example: the teaching part uses a helpful example or analogy
- terms: important technical terms are explained simply

Return ONLY JSON like this:
{{"correct": 1, "faithful": 1, "no_contradiction": 1, "clear": 1, "example": 1, "terms": 1}}
"""
    raw = generate_answer(prompt, model=JUDGE_MODEL)
    if raw is None:
        return None
    try:
        return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except json.JSONDecodeError:
        print("Judge returned bad JSON:", raw)
        return None


def failure_type(row):
    # one bucket per bad result, so we know what to fix
    if "abstained" in row:
        return None if row["abstained"] else "should_have_abstained"
    if row["correct"] == 0:
        if NOT_FOUND_TEXT in row["answer"].lower():
            return "missed_answer"
        return "wrong_answer" if row["hit"] else "retrieval_miss"
    if row["faithful"] == 0:
        return "unsupported_claim"
    if row["no_contradiction"] == 0:
        return "teacher_contradicts_paper"
    if row.get("citation_correct") is False:
        return "wrong_citation"
    if not row["hit"]:
        return "check_evidence"   # correct, but the evidence phrase was not in the top 5
    return None


def average(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def percentile(values, p):
    values = sorted(values)
    return values[round(p / 100 * (len(values) - 1))]


def evaluate_one(item):
    start = time.time()
    result = answer_question(item["question"], item["paper"])
    total_time = time.time() - start

    answer = result["answer"] or ""
    chunks = [s["document"] for s in result["sources"]]

    row = {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "answer": answer,
        "total_time": total_time,
        "timings": result.get("timings", {}),
        "expected": item["answer"],
        "retrieved": [
            {"page": s["metadata"].get("page"), "section": s["metadata"].get("section"), "text": s["document"][:300]}
            for s in result["sources"]
        ],
    }

    if not item["evidence"]:   # unanswerable question
        row["abstained"] = NOT_FOUND_TEXT in answer.lower()
        return row

    rank = find_rank(item["evidence"], chunks)
    row["rank"] = rank
    row["hit"] = rank is not None
    row["reciprocal_rank"] = 1 / rank if rank else 0

    if rank is not None:
        evidence_page = result["sources"][rank - 1]["metadata"].get("page")
        row["citation_correct"] = evidence_page in cited_pages(answer)

    scores = judge_answer(item["question"], item["answer"], answer, "\n\n".join(chunks)) if answer.strip() else None
    for key in JUDGE_KEYS:
        row[key] = scores.get(key) if scores else (0 if not answer.strip() else None)

    return row


def main(run_name):
    warm_up()
    test_set = json.loads(TEST_SET_FILE.read_text(encoding="utf-8"))

    rows = []
    for item in test_set:
        row = evaluate_one(item)
        print(f"[{item['id']}/{len(test_set)}] {row['total_time']:.1f}s  correct={row.get('correct')}  {item['question']}")
        rows.append(row)

    answerable = [r for r in rows if "hit" in r]
    times = [r["total_time"] for r in rows]

    summary = {
        "run": run_name,
        "recall@5": average([r["hit"] for r in answerable]),
        "mrr@5": average([r["reciprocal_rank"] for r in answerable]),
        "correctness": average([r["correct"] for r in answerable]),
        "correctness_facts": average([r["correct"] for r in answerable if r["type"] != "explain"]),
        "correctness_explain": average([r["correct"] for r in answerable if r["type"] == "explain"]),
        "faithfulness": average([r["faithful"] for r in answerable]),
        "no_contradiction": average([r["no_contradiction"] for r in answerable]),
        "teach_clear": average([r["clear"] for r in answerable]),
        "teach_example": average([r["example"] for r in answerable]),
        "teach_terms": average([r["terms"] for r in answerable]),
        "abstain_accuracy": average([r["abstained"] for r in rows if "abstained" in r]),
        "citation_accuracy": average([r["citation_correct"] for r in answerable if "citation_correct" in r]),
        "latency_p50": percentile(times, 50),
        "latency_p95": percentile(times, 95),
    }
    for step in ["rewrite", "retrieve", "rerank", "llm"]:
        summary[f"avg_{step}_time"] = average([r["timings"].get(step) for r in rows])

    print("\n========== SUMMARY ==========")
    for key, value in summary.items():
        print(f"{key:20} {round(value, 3) if isinstance(value, float) else value}")

    failures = [{"failure": failure_type(r), **r} for r in rows if failure_type(r)]
    print(f"\n========== FAILURES ({len(failures)}) ==========")
    for f in failures:
        print(f"Q{f['id']:>2}  {f['failure']:26} {f['question']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{run_name}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=4, ensure_ascii=False), encoding="utf-8")
    (RESULTS_DIR / f"{run_name}_failures.json").write_text(
        json.dumps(failures, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to: {RESULTS_DIR / run_name}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run")
