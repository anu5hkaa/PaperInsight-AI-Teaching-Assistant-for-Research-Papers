import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pdf_processing

from extractor import extract_text
from chunking.chunking import create_chunks
from embeddings.embedding import generate_embeddings
from vectordb.chroma_setup import insert_embeddings, list_papers, paper_exists

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"


def ingest_pdf(pdf_bytes, filename):
    # PDF -> text -> chunks -> embeddings -> Chroma
    paper_name = Path(filename).stem.lower().strip()
    result = {"paper_name": paper_name, "chunks": 0, "inserted": 0, "already_existed": False, "error": None}

    if paper_exists(paper_name):
        result["already_existed"] = True
        return result

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path = UPLOAD_DIR / filename
        pdf_path.write_bytes(pdf_bytes)

        text_file, text = extract_text(pdf_path)
        if not text.strip():
            result["error"] = "Could not extract text from PDF."
            return result

        chunk_file, chunks = create_chunks(text_file)
        embedding_file, _ = generate_embeddings(chunk_file)
        inserted, total = insert_embeddings(embedding_file)

        print(f"{paper_name}: {len(chunks)} chunks, {total} chunks in DB")
        result["chunks"] = len(chunks)
        result["inserted"] = inserted

    except Exception as e:
        result["error"] = str(e)

    return result


if __name__ == "__main__":
    # load every PDF in the data folder
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        print(ingest_pdf(pdf_path.read_bytes(), pdf_path.name))
    print("Papers in DB:", list_papers())
