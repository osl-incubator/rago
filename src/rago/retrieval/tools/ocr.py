"""OCR tools."""

from __future__ import annotations

from pathlib import Path

from rago._optional import require_dependency


def extract_text_from_image(
    file_path: str | Path,
    engine: str = 'tesseract',
) -> str:
    """Extract text from an image using the configured OCR engine."""
    if engine != 'tesseract':
        raise ValueError(
            f'Unsupported OCR engine: {engine}. '
            "Currently supported engines: ['tesseract']"
        )

    pytesseract = require_dependency(
        'pytesseract',
        extra='ocr',
        context='Tesseract OCR',
    )

    return pytesseract.image_to_string(str(file_path))