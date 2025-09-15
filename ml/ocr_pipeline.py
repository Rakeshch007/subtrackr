import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import pdfplumber
import os

def extract_text_from_pdf(pdf_path):
    """
    Try text extraction with pdfplumber first.
    If fails (empty text), fallback to OCR.
    """
    text = ""

    # Try pdfplumber (for text-based PDFs)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print("pdfplumber failed:", e)

    # If text is still empty, fallback to OCR
    if not text.strip():
        print("Falling back to OCR...")
        pages = convert_from_path(pdf_path, 300)
        for page in pages:
            text += pytesseract.image_to_string(page)

    return text.strip()


if __name__ == "__main__":
    sample = os.path.join("data", "sample_statement.pdf")
    if os.path.exists(sample):
        print(extract_text_from_pdf(sample))
    else:
        print("No sample PDF found. Please add one in the data/ folder.")
