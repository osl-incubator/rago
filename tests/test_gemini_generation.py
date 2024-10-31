"""Tests for Rago package using Google's Gemini Model."""

from pathlib import Path

import pytest

from rago import Rago
from rago.augmented import GeminiAug
from rago.generation.gemini_ai import GeminiAIGen
from rago.retrieval import StringRet


@pytest.fixture
def animals_data() -> list[str]:
    """Fixture for loading the animals dataset."""
    data_path = Path(__file__).parent / 'data' / 'animals.txt'
    with open(data_path) as f:
        data = [line.strip() for line in f.readlines() if line.strip()]
    return data


@pytest.fixture
def gemini_api_key() -> str:
    """Fixture for Gemini API key from environment."""
    api_key = 'AIzaSyDl6-k1hPjrgWR71Qia47F6z5dSylYJGpg'
    if not api_key:
        raise EnvironmentError(
            'Please set the GEMINI_API_KEY environment variable.'
        )
    return api_key


@pytest.mark.skip_on_ci
def test_gemini_generation(
    animals_data: list[str], gemini_api_key: str
) -> None:
    """Test RAG pipeline with Gemini model for text generation."""
    # Instantiate Rago with the Gemini model
    rag = Rago(
        retrieval=StringRet(animals_data),
        augmented=GeminiAug(
            k=3
        ),  # Update if using a specific augmentation class for Gemini
        generation=GeminiAIGen(
            api_key=gemini_api_key, model_name='gemini-1.5-flash'
        ),
    )

    query = 'Is there any animal larger than a dinosaur?'
    result = rag.prompt(query)

    # Verify the result contains relevant information, e.g., "Blue Whale"
    assert (
        'Blue Whale' in result
    ), 'Expected response to mention Blue Whale as a larger animal.'
