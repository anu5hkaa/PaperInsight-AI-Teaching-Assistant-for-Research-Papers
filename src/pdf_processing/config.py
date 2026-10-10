# All settings in one place. Change a value, run the eval, compare.

# models
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ANSWER_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-120b"

# speed
REASONING_EFFORT = None    # "low" is faster, only tested on simple questions
MAX_ANSWER_TOKENS = 800    # Groq reserves this per request

# retrieval
VECTOR_TOP_K = 20
BM25_TOP_K = 20
USE_HYBRID = True
USE_RERANKER = False       # +0.02 MRR but 24x slower on my laptop
RERANK_CANDIDATES = 10
FINAL_TOP_K = 5            # chunks sent to the LLM
USE_QUERY_REWRITE = False

# comparison
COMPARE_CHUNKS_FOCUSED = 4
COMPARE_CHUNKS_PER_ASPECT = 2
COMPARE_MAX_CHUNKS_PER_PAPER = 8

# summary
SUMMARY_CHUNKS_PER_ASPECT = 2
SUMMARY_MAX_CHUNKS = 12    # about 2,500 tokens, fits the free-tier limit
