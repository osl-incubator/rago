"""Test image OCR retrieval."""

from pathlib import Path
from unittest.mock import patch

import pytest

from rago.retrieval import ImagePathRet

OCR_DATA_PATH = Path(__file__).parent / 'data' / 'ocr'


def test_image_retrieval_ocr() -> None:
    """Test OCR extraction from an image."""
    image_ret = ImagePathRet(OCR_DATA_PATH / 'test.png')

    with patch(
        'rago.retrieval.file.extract_text_from_image',
        return_value='Hello from OCR',
    ):
        chunks = image_ret.get()

    assert chunks
    assert 'Hello from OCR' in chunks[0]


def test_image_retrieval_rejects_unsupported_file() -> None:
    """Test that unsupported image formats are rejected."""
    unsupported_file = OCR_DATA_PATH / 'unsupported.txt'
    unsupported_file.write_text('not an image')

    try:
        with pytest.raises(Exception, match='supported image format'):
            ImagePathRet(unsupported_file)
    finally:
        unsupported_file.unlink()


def test_image_retrieval_missing_file() -> None:
    """Test that a missing image raises an error."""
    with pytest.raises(Exception, match="File doesn't exist"):
        ImagePathRet(OCR_DATA_PATH / 'missing.png')
