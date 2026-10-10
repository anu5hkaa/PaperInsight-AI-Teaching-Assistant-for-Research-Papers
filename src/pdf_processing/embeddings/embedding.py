import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def generate_embeddings(chunk_file):
    chunk_file = Path(chunk_file)
    paper_name = chunk_file.stem.replace("_chunks", "")
    chunks = json.loads(chunk_file.read_text(encoding="utf-8"))

    # batch encoding is much faster; bge works with normalized vectors
    vectors = get_model().encode(
        [chunk["text"] for chunk in chunks],
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = [
        {
            "chunk_id": chunk["chunk_id"],
            "paper_name": chunk["paper_name"],
            "page": chunk["page"],
            "section": chunk["section"],
            "text": chunk["text"],
            "embedding": vector.tolist(),
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    embedding_file = EMBEDDING_DIR / f"{paper_name}_embeddings.json"
    embedding_file.write_text(json.dumps(embeddings, ensure_ascii=False), encoding="utf-8")

    return embedding_file, embeddings
