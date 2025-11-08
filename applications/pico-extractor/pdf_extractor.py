import os
import pymupdf  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text content from a text-based PDF file.

    Args:
        pdf_path: The absolute or relative path to the PDF file.

    Returns:
        A string containing all the extracted text from the PDF.
        Returns an empty string if the PDF contains no text.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file was not found at path: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    all_text = ""
    for page in doc:
        all_text += page.get_text("text")  # type: ignore
        all_text += "\\n\\n"  # Add space between pages

    doc.close()

    if not all_text.strip():
        return ""

    return all_text
