import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pdf_processing

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from retriever import warm_up
from qa import answer_question
from summarizer import summarize_paper
from comparision import compare_papers
from pipeline.ingest import ingest_pdf, list_papers
from tracing import save_feedback

app = FastAPI(title="PaperInsight API")

warm_up()   # load models once at start, not on the first question


class QuestionRequest(BaseModel):
    question: str
    paper_name: str


class SummaryRequest(BaseModel):
    paper_name: str


class ComparisonRequest(BaseModel):
    question: str
    paper_a: str
    paper_b: str
    paper_c: str | None = None   # optional third paper
    full: bool = False           # compare all main aspects


class FeedbackRequest(BaseModel):
    run_id: str
    score: int                  # 1 = helpful, 0 = not helpful
    comment: str | None = None


@app.get("/")
def home():
    return {"message": "PaperInsight API running"}


@app.post("/upload")
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = ingest_pdf(pdf_bytes, file.filename)
    if result["error"]:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.get("/papers")
def get_papers():
    return {"papers": list_papers()}


@app.post("/ask")
def ask(request: QuestionRequest):
    return answer_question(request.question, request.paper_name)


@app.post("/summary")
def summary(request: SummaryRequest):
    return summarize_paper(request.paper_name)


@app.post("/compare")
def compare(request: ComparisonRequest):
    papers = [p for p in [request.paper_a, request.paper_b, request.paper_c] if p]
    if len(set(papers)) != len(papers):
        raise HTTPException(status_code=400, detail="Please choose different papers.")

    result = compare_papers(request.question, papers, full=request.full)
    return {"comparision": result["comparison"], **result}   # key kept for the UI


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    save_feedback(request.run_id, request.score, request.comment)
    return {"saved": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
