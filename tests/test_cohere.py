"""Tests for Rago package using Cohere AI's API for text gen and aug."""

import os

from typing import cast

import pytest

from rago import Rago
from rago.augmented import CohereAug
from rago.generation import CohereGen
from rago.retrieval import StringRet

from .models import AnimalModel


@pytest.fixture
def api_key(env) -> str:
    """Fixture for Cohere API key from environment."""
    key = os.getenv('COHERE_API_KEY')
    if not key:
        raise EnvironmentError(
            'Please set the COHERE_API_KEY environment variable.'
        )
    return key


@pytest.mark.skip_on_ci
def test_cohere_generation(animals_data: list[str], api_key: str) -> None:
    """Test RAG pipeline with Cohere model for text generation."""
    logs = {
        'retrieval': {},
        'augmented': {},
        'generation': {},
    }
    rag = Rago(
        retrieval=StringRet(animals_data, logs=logs['retrieval']),
        augmented=CohereAug(top_k=3, logs=logs['augmented'], api_key=api_key),
        generation=CohereGen(
            api_key=api_key,
            model_name='command-r-plus-08-2024',
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
def test_rag_cohere_structured_output(
    api_key: str,
    animals_data: list[str],
    question: str,
    expected_answer: str,
) -> None:
    """Test RAG pipeline with Cohere for structured output generation."""
    logs = {
        'retrieval': {},
        'augmented': {},
        'generation': {},
    }
    rag = Rago(
        retrieval=StringRet(animals_data, logs=logs['retrieval']),
        augmented=CohereAug(top_k=3, logs=logs['augmented'], api_key=api_key),
        generation=CohereGen(
            api_key=api_key,
            model_name='command-r-plus-08-2024',
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
