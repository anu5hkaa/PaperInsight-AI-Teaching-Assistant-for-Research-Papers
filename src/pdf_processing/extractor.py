from pathlib import Path
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"

# put before every page so the chunker knows the page number
PAGE_MARKER = "<<<PAGE {}>>>"


def extract_text(pdf_path, output_dir=EXTRACTED_DIR):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    pages = [
        PAGE_MARKER.format(number) + "\n" + (page.extract_text() or "")
        for number, page in enumerate(reader.pages, start=1)
    ]
    text = "\n".join(pages)

    output_path = output_dir / f"{pdf_path.stem.lower().strip()}.txt"
    output_path.write_text(text, encoding="utf-8")

    return output_path, text
