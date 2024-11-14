"""PDF tools."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def is_pdf(file_path: str | Path) -> bool:
    """
    Check if a file is a PDF by reading its header.

    Parameters
    ----------
    file_path : str
        Path to the file to be checked.

    Returns
    -------
    bool
        True if the file is a PDF, False otherwise.
    """
    try:
        with open(file_path, 'rb') as file:
            header = file.read(4)
            return header == b'%PDF'
    except IOError:
        return False


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using pypdf.

    The result is the same as the one returned by PyPDFLoader.
    """
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return ' '.join(pages)
