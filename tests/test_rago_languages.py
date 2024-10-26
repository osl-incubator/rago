"""Tests for language detection and response generation in Rago."""

import os

import pytest

from rago import Rago
from rago.augmented import HuggingFaceAug
from rago.generation import LlamaV32M1BGen
from rago.retrieval import StringRet


@pytest.fixture
def animals_data_en() -> list[str]:
    """Create a fixture with a shorter animals dataset in English."""
    return [
        'Blue Whale: The Blue Whale is the largest animal in existence.',
        'Peregrine Falcon: Known as the fastest animal on Earth.',
    ]


@pytest.fixture
def animals_data_fr() -> list[str]:
    """Create a fixture with a shorter animals dataset in French."""
    return [
        'Baleine bleue: La baleine bleue est le plus grand animal qui existe.',
        "Faucon pèlerin: Connu comme l'animal le plus rapide sur Terre.",
    ]


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
    animals_data_en: list[str],
    animals_data_fr: list[str],
    device: str = 'cpu',
) -> None:
    """Test language detection and response generation in Rago."""
    HF_TOKEN = os.getenv('HF_TOKEN', '')
    animals_data = (
        animals_data_en if expected_language == 'en' else animals_data_fr
    )

    rag = Rago(
        retrieval=StringRet(animals_data),
        augmented=HuggingFaceAug(k=3),
        generation=LlamaV32M1BGen(apikey=HF_TOKEN, device=device),
    )

    detected_language = rag.detect_language(query)
    assert (
        detected_language == expected_language
    ), f'Expected {expected_language}, got {detected_language}'

    response = rag.prompt(query)
    assert isinstance(response, str), 'Response should be a string.'
    assert response != '', 'Response should not be empty.'
