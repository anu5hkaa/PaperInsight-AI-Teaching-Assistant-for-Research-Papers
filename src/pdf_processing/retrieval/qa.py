import time

from retriever import hybrid_search, rerank_chunks   # first: sets up the import path
from config import USE_RERANKER, RERANK_CANDIDATES, FINAL_TOP_K, USE_QUERY_REWRITE, REASONING_EFFORT
from promt_builder import build_qa_prompt
from generator import generate_answer
from tracing import traceable, current_run_id

NOT_FOUND = "The provided context does not contain enough information."

_answer_cache = {}   # (question, paper_name) -> result


def rewrite_query(question):
    prompt = (
        "Rewrite this question about a research paper as a short search query.\n"
        "Keep all technical terms and numbers. Return only the query.\n\n"
        f"Question: {question}"
    )
    return generate_answer(prompt) or question


@traceable(name="answer_question")
def answer_question(question, paper_name):
    cache_key = (question.strip().lower(), paper_name)
    if cache_key in _answer_cache:
        return {**_answer_cache[cache_key], "run_id": current_run_id()}

    timings = {}

    search_query = question
    if USE_QUERY_REWRITE:
        t0 = time.time()
        search_query = rewrite_query(question)
        timings["rewrite"] = time.time() - t0

    # 1. hybrid search
    t0 = time.time()
    candidates = hybrid_search(search_query, paper_name)
    timings["retrieve"] = time.time() - t0

    if not candidates:
        return {"paper_name": paper_name, "answer": NOT_FOUND, "sources": [], "metadata": [], "timings": timings}

    # 2. optional rerank
    if USE_RERANKER:
        t0 = time.time()
        candidates = candidates[:RERANK_CANDIDATES]
        meta_of = dict(candidates)
        reranked = rerank_chunks(question, [doc for doc, _ in candidates])
        top_docs = [
            {"document": doc, "metadata": meta_of[doc], "score": float(score)}
            for score, doc in reranked[:FINAL_TOP_K]
        ]
        timings["rerank"] = time.time() - t0
    else:
        top_docs = [{"document": doc, "metadata": meta, "score": None} for doc, meta in candidates[:FINAL_TOP_K]]

    # 3. answer
    context = "\n\n".join(item["document"] for item in top_docs)
    t0 = time.time()
    answer = generate_answer(build_qa_prompt(question, context), reasoning_effort=REASONING_EFFORT)
    timings["llm"] = time.time() - t0

    result = {
        "paper_name": paper_name,
        "answer": answer,
        "sources": top_docs,
        "metadata": [item["metadata"] for item in top_docs],
        "timings": timings,
        "run_id": current_run_id(),
    }
    if answer:
        _answer_cache[cache_key] = result
    return result
