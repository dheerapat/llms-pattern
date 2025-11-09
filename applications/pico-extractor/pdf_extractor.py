import os
import pymupdf


def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file was not found at path: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    all_text = ""
    for page in doc:
        all_text += page.get_text("text")  # pyright: ignore
        all_text += "\n\n"

    doc.close()

    if not all_text.strip():
        return ""

    return all_text
