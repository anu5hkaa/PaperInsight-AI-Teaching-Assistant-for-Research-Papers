import re
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

CHUNK_WORDS = 150        # about 200 tokens
OVERLAP_SENTENCES = 2    # repeated at the start of the next chunk

# "3.2 Positional Encoding", "4 Why Self-Attention", "A Appendix"
NUMBERED_HEADING = re.compile(r"^(\d{1,2}(\.\d{1,2}){0,2}|[A-C](\.\d{1,2})?)\s+[A-Z][A-Za-z\-:, ]{2,60}$")
NAMED_HEADINGS = {"abstract", "introduction", "conclusion", "conclusions", "references", "acknowledgements"}
PAGE_LINE = re.compile(r"^<<<PAGE (\d+)>>>$")


def is_heading(line):
    if len(line.split()) > 8:
        return False
    if line.lower() in NAMED_HEADINGS:
        return True
    return bool(NUMBERED_HEADING.match(line))


def split_sentences(text):
    # split after . ? ! when the next word starts with a capital or a digit
    return [s for s in re.split(r"(?<=[.?!])\s+(?=[A-Z0-9(])", text) if s.strip()]


def read_sentences(text):
    # returns (sentence, page, section); skips the References list
    sentences = []
    page = 1
    section = "Title"
    buffer = []

    def flush():
        block = " ".join(" ".join(buffer).split())
        if section != "References":
            for sentence in split_sentences(block):
                words = sentence.split()
                # tables have no full stops, so cut very long "sentences"
                for i in range(0, len(words), CHUNK_WORDS):
                    sentences.append((" ".join(words[i:i + CHUNK_WORDS]), page, section))
        buffer.clear()

    for line in text.split("\n"):
        line = line.strip()

        page_match = PAGE_LINE.match(line)
        if page_match:
            flush()
            page = int(page_match.group(1))
            continue

        if is_heading(line):
            flush()
            section = "References" if line.lower() == "references" else line
            continue

        buffer.append(line)

    flush()
    return sentences


def create_chunks(text_file, output_dir=CHUNKS_DIR):
    # section-aware, sentence-based chunks with page + section
    text_file = Path(text_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_name = text_file.stem
    sentences = read_sentences(text_file.read_text(encoding="utf-8"))

    chunks = []
    current = []

    def save_chunk():
        body = " ".join(s for s, _, _ in current)
        page, section = current[0][1], current[0][2]
        chunks.append({
            "chunk_id": len(chunks),
            "paper_name": paper_name,
            "page": page,
            "section": section,
            "text": f"[{paper_name} | {section} | page {page}] {body}",
        })

    for sentence in sentences:
        # a chunk never mixes two sections
        if current and sentence[2] != current[-1][2]:
            save_chunk()
            current = []

        current.append(sentence)

        if sum(len(s.split()) for s, _, _ in current) >= CHUNK_WORDS:
            save_chunk()
            current = current[-OVERLAP_SENTENCES:]

    # leftover, unless it is only the overlap of the last chunk
    if current and (not chunks or len(current) > OVERLAP_SENTENCES):
        save_chunk()

    output_path = output_dir / f"{paper_name}_chunks.json"
    output_path.write_text(json.dumps(chunks, indent=4, ensure_ascii=False), encoding="utf-8")

    return output_path, chunks
