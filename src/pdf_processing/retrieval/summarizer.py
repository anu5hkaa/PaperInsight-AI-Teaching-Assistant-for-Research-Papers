import time

from retriever import search_aspects   # first: sets up the import path
from config import SUMMARY_CHUNKS_PER_ASPECT, SUMMARY_MAX_CHUNKS
from promt_builder import build_summary_prompt
from generator import generate_answer
from tracing import traceable, current_run_id

# the parts every summary should cover
SUMMARY_ASPECTS = [
    "abstract and main contribution of the paper",
    "problem and motivation",
    "proposed method and model architecture",
    "training data and training setup",
    "main results and evaluation numbers",
    "limitations, conclusion and future work",
]


@traceable(name="summarize_paper")
def summarize_paper(paper_name):
    timings = {}

    # 1. find the most important chunks for each part of the summary
    t0 = time.time()
    chunks = search_aspects(SUMMARY_ASPECTS, paper_name, SUMMARY_CHUNKS_PER_ASPECT, SUMMARY_MAX_CHUNKS)
    timings["retrieve"] = time.time() - t0

    # keep the paper's own order, so the context reads like the paper
    chunks.sort(key=lambda item: item[1].get("chunk_id", 0))
    context = "\n\n".join(doc for doc, _ in chunks)

    # 2. one LLM call for the whole summary
    t0 = time.time()
    summary = generate_answer(build_summary_prompt(paper_name, context))
    timings["llm"] = time.time() - t0

    return {
        "paper_name": paper_name,
        "summary": summary or "Summary generation failed. Please try again.",
        "chunks_used": len(chunks),
        "sources": [{"document": doc, "metadata": meta} for doc, meta in chunks],
        "timings": timings,
        "run_id": current_run_id(),
    }
