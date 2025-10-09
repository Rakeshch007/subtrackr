import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import pdfplumber
import os

from pathlib import Path

def extract_text_from_file(path):
    """
    Extract text from a file.
    Supports:
      - PDFs (text or scanned)
      - Images (PNG, JPG, etc.)
    """
    ext = Path(path).suffix.lower()
    text = ""

    if ext == ".pdf":
        # Try text-based extraction first
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception as e:
            print("pdfplumber failed:", e)

        # If empty, fallback to OCR
        if not text.strip():
            print("Falling back to OCR...")
            pages = convert_from_path(path, 300)
            for page in pages:
                text += pytesseract.image_to_string(page)

    elif ext in [".png", ".jpg", ".jpeg"]:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return text.strip()



    


if __name__ == "__main__":
    sample = os.path.join("data", "temp.jpg")
    if os.path.exists(sample):
        print(extract_text_from_file(sample))
    else:
        print("No sample file found. Please add one in the data/ folder.")