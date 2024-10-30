"""Tests for Rago package using OpenAI GPT-4."""

import os

from pathlib import Path

import pytest

from rago import Rago
from rago.augmented import OpenAIAug
from rago.generation.openai_gpt import (
    OpenAIGPTGen,
)
from rago.retrieval import StringRet


@pytest.fixture
def animals_data() -> list[str]:
    """Fixture for loading the animals dataset."""
    data_path = Path(__file__).parent / 'data' / 'animals.txt'
    with open(data_path) as f:
        data = [line.strip() for line in f.readlines() if line.strip()]
        return data


@pytest.fixture
def openai_api_key() -> str:
    """Fixture for OpenAI API key from environment."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError(
            'Please set the OPENAI_API_KEY environment variable.'
        )
    return api_key


@pytest.mark.skip_on_ci
def test_openai_gpt4(animals_data: list[str], openai_api_key: str) -> None:
    """Test RAG pipeline with OpenAI's GPT-4."""
    rag = Rago(
        retrieval=StringRet(animals_data),
        augmented=OpenAIAug(k=3),
        generation=OpenAIGPTGen(api_key=openai_api_key, model_name='gpt-4'),
    )

    query = 'Is there any animal larger than a dinosaur?'
    result = rag.prompt(query)

    assert (
        'Blue Whale' in result
    ), 'Expected response to mention Blue Whale as a larger animal.'
