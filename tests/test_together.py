"""Tests for Rago package using Together Model."""

import os

from typing import cast

import pytest

from rago import Rago
from rago.augmented import SentenceTransformerAug
from rago.generation import TogetherGen
from rago.retrieval import StringRet

from .models import AnimalModel


@pytest.fixture
def api_key(env) -> str:
    """Fixture for Together API key from environment."""
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        raise EnvironmentError(
            'Please set the TOGETHER_API_KEY environment variable.'
        )
    return api_key


@pytest.mark.skip_on_ci
def test_together_generation(animals_data: list[str], api_key: str) -> None:
    """Test RAG pipeline with Together model for text generation."""
    # Instantiate Rago with the Together model
    logs = {
        'retrieval': {},
        'augmented': {},
        'generation': {},
    }
    rag = Rago(
        retrieval=StringRet(animals_data, logs=logs['retrieval']),
        augmented=SentenceTransformerAug(
            top_k=3, logs=logs['augmented'], api_key=api_key
        ),
        generation=TogetherGen(
            api_key=api_key,
            model_name='meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
            logs=logs['generation'],
        ),
    )

    query = 'Is there any animal larger than a dinosaur?'
    result = rag.prompt(query)

    # Verify the result contains relevant information, e.g., "Blue Whale"
    assert 'blue whale' in result.lower(), (
        'Expected response to mention Blue Whale as a larger animal.'
    )

    # check if logs have been used
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
def test_rag_together_structured_output(
    api_key: str,
    animals_data: list[str],
    question: str,
    expected_answer: str,
) -> None:
    """Test RAG pipeline with Together for structured output."""
    logs = {
        'retrieval': {},
        'augmented': {},
        'generation': {},
    }

    rag = Rago(
        retrieval=StringRet(animals_data, logs=logs['retrieval']),
        augmented=SentenceTransformerAug(
            top_k=3, logs=logs['augmented'], api_key=api_key
        ),
        generation=TogetherGen(
            api_key=api_key,
            model_name='meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
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

    # check if logs have been used
    assert logs['retrieval']
    assert logs['augmented']
    assert logs['generation']
