import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pdf_processing

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, VECTOR_TOP_K, BM25_TOP_K, USE_HYBRID, USE_RERANKER
from vectordb.chroma_setup import collection, list_papers
from embeddings.embedding import get_model as get_embedding_model
from tracing import traceable

# official bge prefix for search queries
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_reranker = None
_bm25_cache = {}   # paper_name -> (bm25, documents, metadatas)


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def retrieve_complete_paper(paper_name):
    return collection.get(where={"paper_name": paper_name}, include=["documents", "metadatas"])


# vector search: similar meaning
def retrieve_chunks(question, paper_name, n_results=VECTOR_TOP_K):
    embedding = get_embedding_model().encode(QUERY_PREFIX + question, normalize_embeddings=True)
    return collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=n_results,
        where={"paper_name": paper_name},
        include=["documents", "metadatas"],
    )


# keyword search: exact words like "WordPiece" or "28.4"
def tokenize(text):
    return re.findall(r"\w+", text.lower())


def get_bm25(paper_name):
    # built once per paper, then reused
    if paper_name not in _bm25_cache:
        results = retrieve_complete_paper(paper_name)
        bm25 = BM25Okapi([tokenize(doc) for doc in results["documents"]])
        _bm25_cache[paper_name] = (bm25, results["documents"], results["metadatas"])
    return _bm25_cache[paper_name]


def bm25_search(question, paper_name, n_results=BM25_TOP_K):
    bm25, documents, metadatas = get_bm25(paper_name)
    scores = bm25.get_scores(tokenize(question))
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    return [(documents[i], metadatas[i]) for i in best]


def reciprocal_rank_fusion(result_lists, k=60):
    # each list gives a chunk 1 / (k + rank); chunks high in both lists win
    scores, metas = {}, {}
    for results in result_lists:
        for rank, (doc, meta) in enumerate(results, start=1):
            scores[doc] = scores.get(doc, 0) + 1 / (k + rank)
            metas[doc] = meta
    return [(doc, metas[doc]) for doc in sorted(scores, key=scores.get, reverse=True)]


@traceable(name="hybrid_search", run_type="retriever")
def hybrid_search(question, paper_name):
    # returns [(document, metadata)], best first
    results = retrieve_chunks(question, paper_name)
    vector_list = list(zip(results["documents"][0], results["metadatas"][0]))

    if not USE_HYBRID:
        return vector_list

    return reciprocal_rank_fusion([vector_list, bm25_search(question, paper_name)])


@traceable(name="search_aspects", run_type="retriever")
def search_aspects(aspects, paper_name, chunks_per_aspect, max_chunks):
    # hybrid search for several aspects in one paper: one embedding batch, one Chroma query
    vectors = get_embedding_model().encode([QUERY_PREFIX + a for a in aspects], normalize_embeddings=True)
    results = collection.query(
        query_embeddings=vectors.tolist(),
        n_results=VECTOR_TOP_K,
        where={"paper_name": paper_name},
        include=["documents", "metadatas"],
    )

    chosen, seen = [], set()

    for i, aspect in enumerate(aspects):
        vector_list = list(zip(results["documents"][i], results["metadatas"][i]))
        ranked = reciprocal_rank_fusion([vector_list, bm25_search(aspect, paper_name)]) if USE_HYBRID else vector_list

        added = 0
        for doc, meta in ranked:
            if doc not in seen:
                seen.add(doc)
                chosen.append((doc, meta))
                added += 1
            if added == chunks_per_aspect:
                break

    return chosen[:max_chunks]


def rerank_chunks(question, documents):
    if not documents:
        return []
    scores = get_reranker().predict([(question, doc) for doc in documents])
    return sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)


def warm_up():
    # load models and keyword indexes before the first question
    get_embedding_model().encode("warm up", normalize_embeddings=True)
    for paper_name in list_papers():
        get_bm25(paper_name)
    if USE_RERANKER:
        get_reranker()
    print("Models loaded and keyword indexes ready.")
