import os
import pdfplumber
import docx


def extract_text_from_file(file_path):
    """Extracts plain text from uploaded PDF, DOCX, or TXT files."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext in [".docx", ".doc"]:
        return _extract_docx(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _extract_pdf(file_path):
    text_list = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text_list.append(extracted)
    return " ".join(text_list)


def _extract_docx(file_path):
    doc = docx.Document(file_path)
    return " ".join([para.text for para in doc.paragraphs if para.text])
