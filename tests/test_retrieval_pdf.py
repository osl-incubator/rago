"""Test the PDF retrieval."""

from pathlib import Path

from rago.retrieval import PDFPathRet

PDF_DATA_PATH = Path(__file__).parent / 'data' / 'pdf'


def test_retrieval_pdf_extraction() -> None:
    """Test the text extraction from a pdf."""
    pdf_ret = PDFPathRet(PDF_DATA_PATH / '1.pdf')
    chunks = pdf_ret.get()

    assert len(chunks) >= 100
