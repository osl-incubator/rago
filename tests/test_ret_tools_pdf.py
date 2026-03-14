"""Test the PDF retrieval."""

from functools import partial
from pathlib import Path

import pytest

from rago.augmented import Augmented
from rago.retrieval import Retrieval

from tests.helpers import call_or_skip, require_spacy_model

PDF_DATA_PATH = Path(__file__).parent / 'data' / 'pdf'


SpaCyAug = partial(Augmented, backend='spacy', model_name='en_core_web_md')
PDFPathRet = partial(Retrieval, backend='pdf')


def test_retrieval_pdf_extraction_basic() -> None:
    """Test the text extraction from a pdf."""
    pdf_ret = PDFPathRet(PDF_DATA_PATH / '1.pdf')
    chunks = pdf_ret.get()

    assert len(chunks) >= 100


@pytest.mark.parametrize(
    'pdf_path,expected',
    [
        ('2407.13797.pdf', ''),
        ('2407.20116.pdf', ''),
    ],
)
def test_retrieval_pdfs_extraction_aug_spacy(
    pdf_path: str, expected: str
) -> None:
    """Test the text extraction from a pdf."""
    del expected
    pdf_ret = PDFPathRet(PDF_DATA_PATH / pdf_path)
    chunks = pdf_ret.get()

    min_total_chunks = 100  # arbitrary number
    max_chunk_size = pdf_ret.splitter.chunk_size

    assert len(chunks) >= min_total_chunks
    for chunk in chunks:
        assert len(chunk) < max_chunk_size

    query = 'What are the key barriers to implementing vitamin D?'

    aug_top_k = 3

    require_spacy_model('en_core_web_md')
    aug_openai = call_or_skip('spacy', SpaCyAug, top_k=aug_top_k)
    aug_result = call_or_skip(
        'spacy',
        aug_openai.search,
        query,
        documents=chunks,
    )

    assert aug_result
    assert len(aug_result) == aug_top_k
    assert all(['vitamin' in result.lower() for result in aug_result])
    assert all(['vitamin d' in result.lower() for result in aug_result])
    assert len(set(aug_result)) == 3
