# Offline eval in LangSmith: same test set and metrics as evaluate.py,
# every question becomes a trace and every run an experiment.
# Needs in .env: LANGSMITH_API_KEY=... and LANGSMITH_TRACING=true
# Usage: python src/pdf_processing/evaluation/langsmith_eval.py teacher_v3
import os
import sys
import json
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "retrieval"))

from qa import answer_question
from retriever import warm_up
from langsmith import Client

# reuse the metric code we already wrote, so both evals score the same way
from evaluate import judge_answer, find_rank, cited_pages, NOT_FOUND_TEXT, JUDGE_KEYS

os.environ["LANGSMITH_TRACING"] = "true"   # evaluate.py turns it off; experiments need it on

DATASET_NAME = "paperinsight-qa"
TEST_SET_FILE = EVAL_DIR / "test_set.json"
PAUSE_SECONDS = 10   # like a real student: one question, then some reading time

client = Client()


def upload_dataset():
    # create the dataset once; later runs reuse it
    if client.has_dataset(dataset_name=DATASET_NAME):
        return

    with open(TEST_SET_FILE, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    dataset = client.create_dataset(dataset_name=DATASET_NAME, description="PaperInsight Q&A test set")

    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": q["question"], "paper": q["paper"]},
                "outputs": {"answer": q["answer"], "evidence": q["evidence"], "type": q["type"]},
            }
            for q in test_set
        ],
    )
    print(f"Uploaded {len(test_set)} examples to LangSmith dataset '{DATASET_NAME}'")


def target(inputs):
    # the system we are testing: question in, answer + retrieved chunks out
    time.sleep(PAUSE_SECONDS)   # not counted in latency

    start = time.time()
    result = answer_question(inputs["question"], inputs["paper"])
    latency = time.time() - start

    return {
        "answer": result["answer"] or "",
        "chunks": [s["document"] for s in result["sources"]],
        "pages": [s["metadata"].get("page") for s in result["sources"]],
        "latency": latency,
    }


# ---------------- evaluators (inputs, outputs, reference_outputs) ----------------

def retrieval(inputs, outputs, reference_outputs):
    if not reference_outputs["evidence"]:   # unanswerable: nothing to find
        return {"results": [{"key": "recall@5", "score": None}, {"key": "mrr@5", "score": None}]}
    rank = find_rank(reference_outputs["evidence"], outputs["chunks"])
    return {"results": [
        {"key": "recall@5", "score": int(rank is not None)},
        {"key": "mrr@5", "score": 1 / rank if rank else 0},
    ]}


def llm_judge(inputs, outputs, reference_outputs):
    if not reference_outputs["evidence"]:
        return {"results": [{"key": key, "score": None} for key in JUDGE_KEYS]}
    context = "\n\n".join(outputs["chunks"])
    scores = judge_answer(inputs["question"], reference_outputs["answer"], outputs["answer"], context) or {}
    return {"results": [{"key": key, "score": scores.get(key)} for key in JUDGE_KEYS]}


def abstain(inputs, outputs, reference_outputs):
    if reference_outputs["evidence"]:
        return {"key": "abstain", "score": None}
    return {"key": "abstain", "score": int(NOT_FOUND_TEXT in outputs["answer"].lower())}


def citation(inputs, outputs, reference_outputs):
    rank = find_rank(reference_outputs["evidence"], outputs["chunks"]) if reference_outputs["evidence"] else None
    if rank is None:
        return {"key": "citation", "score": None}
    evidence_page = outputs["pages"][rank - 1]
    return {"key": "citation", "score": int(evidence_page in cited_pages(outputs["answer"]))}


def latency(inputs, outputs, reference_outputs):
    return {"key": "latency_s", "score": round(outputs["latency"], 2)}


if __name__ == "__main__":

    run_name = sys.argv[1] if len(sys.argv) > 1 else "qa"

    warm_up()
    upload_dataset()

    client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[retrieval, llm_judge, abstain, citation, latency],
        experiment_prefix=run_name,
        max_concurrency=1,   # one question at a time (free-tier rate limits)
    )

    print("\nDone. Open LangSmith -> Datasets & Experiments -> paperinsight-qa to see the results.")
