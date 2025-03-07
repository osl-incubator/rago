"""Tests for language detection."""

import pytest

from langdetect import detect


@pytest.mark.parametrize(
    'query,expected_language',
    [
        ('Are there animals bigger than a dinosaur?', 'en'),
        ("Y a-t-il des animaux plus grands qu'un dinosaure?", 'fr'),
    ],
)
def test_language_detection(
    query: str,
    expected_language: str,
) -> None:
    """Test language detection."""
    detected_language = detect(query)
    assert detected_language == expected_language, (
        f'Expected {expected_language}, got {detected_language}'
    )
