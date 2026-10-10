# PaperInsight: AI Teaching Assistant for Research Papers

Upload research papers (PDF) and learn them with an AI that explains like a teacher, while staying grounded in the paper with page citations.

## Features

- **Ask questions**: technical answer + simple explanation + key terms, with page citations
- **Summarize** a paper: problem, method, data, results, limitations
- **Compare 2 or 3 papers**: on one aspect (e.g. methodology) or a full comparison
- **Offline + online LLM evaluation** with LangSmith tracing and user feedback

## Results

| Metric | Before | After |
|---|---|---|
| Correctness | 88% | **96%** |
| Faithfulness (no invented facts) | 54% | **96%** |
| Citation accuracy | - | **100%** |
| Median latency | 21.3 s | **1.4 s** (15x faster) |

Measured on a ground-truth test set with an LLM judge.

## Highlights

- **Hybrid retrieval**: BM25 keyword search + vector search, merged with Reciprocal Rank Fusion
- **Section-aware chunking** with page numbers for citations
- **Data-driven speed-up**: profiled each step and removed a reranker that was 24x slower for almost no gain
- **Evaluation loop**: offline test set (correctness, faithfulness, citations, abstention) + online LangSmith tracing with Helpful / Not helpful feedback

## Tech Stack

Python, FastAPI, Streamlit, ChromaDB, BGE embeddings, BM25, Groq (gpt-oss), LangSmith

## How to Run

1. Install:
   ```
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_key
   LANGSMITH_API_KEY=your_langsmith_key   # optional
   LANGSMITH_TRACING=true                 # optional
   LANGSMITH_PROJECT=paperinsight         # optional
   ```
3. Start the backend:
   ```
   python src/pdf_processing/retrieval/app.py
   ```
4. Start the app (new terminal):
   ```
   python -m streamlit run src/pdf_processing/streamlit_app.py
   ```
5. Open **http://localhost:8501**, upload a PDF, and ask, summarize or compare.

Run the evaluation:
```
python src/pdf_processing/evaluation/evaluate.py my_run
```
