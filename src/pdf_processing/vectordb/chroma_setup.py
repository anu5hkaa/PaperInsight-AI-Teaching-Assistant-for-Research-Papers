import json
from pathlib import Path
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "data" / "chroma_db"

client = chromadb.PersistentClient(path=str(DB_PATH))

# v2 = section-aware chunks + cosine distance
collection = client.get_or_create_collection(
    name="research_papers_v2",
    metadata={"hnsw:space": "cosine"},
)


def insert_embeddings(embedding_file):
    items = json.loads(Path(embedding_file).read_text(encoding="utf-8"))

    # one call for all chunks
    collection.add(
        ids=[f"{item['paper_name']}_{item['chunk_id']}" for item in items],
        embeddings=[item["embedding"] for item in items],
        documents=[item["text"] for item in items],
        metadatas=[
            {
                "chunk_id": item["chunk_id"],
                "paper_name": item["paper_name"],
                "page": item["page"],
                "section": item["section"],
            }
            for item in items
        ],
    )
    return len(items), collection.count()


def paper_exists(paper_name):
    results = collection.get(where={"paper_name": paper_name}, include=["metadatas"])
    return len(results["ids"]) > 0


def list_papers():
    results = collection.get(include=["metadatas"])
    return sorted(set(m["paper_name"] for m in results["metadatas"]))
