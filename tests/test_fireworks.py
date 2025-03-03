"""Tests for Rago package using Fireworks AI for gen and aug."""

import os

from typing import cast

import pytest

from rago import Rago
from rago.augmented import FireworksAug
from rago.generation import FireworksGen
from rago.retrieval import StringRet

from .models import AnimalModel


@pytest.fixture
def api_key(env) -> str:
    """Fixture for Fireworks API key from environment."""
    key = os.getenv('FIREWORKS_API_KEY')
    if not key:
        raise EnvironmentError(
            'Please set the FIREWORKS_API_KEY environment variable.'
        )
    return key


@pytest.mark.skip_on_ci
def test_fireworks_generation(animals_data: list[str], api_key: str) -> None:
    """Test RAG pipeline with Fireworks model for text generation."""
    logs = {
        'retrieval': {},
        'augmented': {},
        'generation': {},
    }
    rag = Rago(
        retrieval=StringRet(animals_data, logs=logs['retrieval']),
        augmented=FireworksAug(
            top_k=3, logs=logs['augmented'], api_key=api_key
        ),
        generation=FireworksGen(
            api_key=api_key,
            model_name='accounts/fireworks/models/llama-v3-8b-instruct',
            logs=logs['generation'],
        ),
    )

    query = 'Is there any animal larger than a dinosaur?'
    result = rag.prompt(query)

    # Verify the result contains relevant information, e.g., "Blue Whale"
    assert 'blue whale' in result.lower(), (
        'Expected response to mention Blue Whale as a larger animal.'
    )

    # Check that logs have been populated
    assert logs['retrieval']
    assert logs['augmented']
    assert logs['generation']


@pytest.mark.skip_on_ci
@pytest.mark.parametrize(
    'question,expected_answer',
    [
        ('What animal is larger than a dinosaur?', 'Blue Whale'),
        (
            'What ave is renowned as the fastest animal on the planet?',
            'Peregrine Falcon',
        ),
    ],
)
def test_rag_fireworks_structured_output(
    api_key: str,
    animals_data: list[str],
    question: str,
    expected_answer: str,
) -> None:
    """Test RAG pipeline with Fireworks for structured output generation."""
    logs = {
        'retrieval': {},
        'augmented': {},
        'generation': {},
    }

    rag = Rago(
        retrieval=StringRet(animals_data, logs=logs['retrieval']),
        augmented=FireworksAug(
            top_k=3, logs=logs['augmented'], api_key=api_key
        ),
        generation=FireworksGen(
            api_key=api_key,
            model_name='accounts/fireworks/models/llama-v3-8b-instruct',
            logs=logs['generation'],
            structured_output=AnimalModel,
        ),
    )

    result = cast(AnimalModel, rag.prompt(question))

    error_message = (
        f'Expected response to mention `{expected_answer}`. '
        f'Result: `{result.name}`.'
    )
    assert expected_answer == result.name, error_message

    # Check that logs have been populated
    assert logs['retrieval']
    assert logs['augmented']
    assert logs['generation']
