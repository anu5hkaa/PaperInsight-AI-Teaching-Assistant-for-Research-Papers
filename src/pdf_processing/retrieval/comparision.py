import time

from retriever import search_aspects
from config import COMPARE_CHUNKS_FOCUSED, COMPARE_CHUNKS_PER_ASPECT, COMPARE_MAX_CHUNKS_PER_PAPER
from promt_builder import build_compare_prompt
from generator import generate_answer
from tracing import traceable, current_run_id

# used when the student asks for a full comparison
DEFAULT_ASPECTS = [
    "main goal and the problem the paper solves",
    "model architecture and method",
    "training data and training setup",
    "main results and evaluation",
    "limitations and future work",
]


@traceable(name="compare_papers")
def compare_papers(question, paper_names, full=False):
    # full=False: compare what the student asked; full=True: all DEFAULT_ASPECTS
    if not 2 <= len(paper_names) <= 3:
        raise ValueError("Please choose 2 or 3 papers.")

    timings = {}

    if full:
        aspects, chunks_per_aspect = DEFAULT_ASPECTS, COMPARE_CHUNKS_PER_ASPECT
    else:
        aspects, chunks_per_aspect = [question], COMPARE_CHUNKS_FOCUSED

    # 1. search every paper separately, so each paper gets its own evidence
    t0 = time.time()
    sources = {
        paper: search_aspects(aspects, paper, chunks_per_aspect, COMPARE_MAX_CHUNKS_PER_PAPER)
        for paper in paper_names
    }
    timings["retrieve"] = time.time() - t0

    # 2. context grouped by paper, so the LLM does not mix them up
    context = "\n\n".join(
        f"===== PAPER: {paper} =====\n" + "\n\n".join(doc for doc, _ in chunks)
        for paper, chunks in sources.items()
    )

    # 3. one LLM call for the whole comparison
    t0 = time.time()
    answer = generate_answer(build_compare_prompt(question, paper_names, aspects, context))
    timings["llm"] = time.time() - t0

    return {
        "comparison": answer,
        "papers": paper_names,
        "aspects": aspects,
        "sources": {
            paper: [{"document": doc, "metadata": meta} for doc, meta in chunks]
            for paper, chunks in sources.items()
        },
        "timings": timings,
        "run_id": current_run_id(),
    }
