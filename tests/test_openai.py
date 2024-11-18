"""Tests for Rago package using OpenAI GPT."""

import os

import pytest

from rago import Rago
from rago.augmented import OpenAIAug
from rago.generation import OpenAIGen
from rago.retrieval import StringRet


@pytest.fixture
def api_key(env) -> str:
    """Fixture for OpenAI API key from environment."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError(
            'Please set the OPENAI_API_KEY environment variable.'
        )
    return api_key


@pytest.mark.skip_on_ci
def test_aug_openai(animals_data: list[str], api_key: str) -> None:
    """Test RAG pipeline with OpenAI's GPT."""
    logs = {
        'augmented': {},
    }

    query = 'Is there any animal larger than a dinosaur?'
    top_k = 3

    ret_string = StringRet(animals_data)
    aug_openai = OpenAIAug(
        api_key=api_key,
        top_k=top_k,
        logs=logs['augmented'],
    )

    ret_result = ret_string.get()
    aug_result = aug_openai.search(query, ret_result)

    assert aug_openai.top_k == top_k
    assert len(aug_result) == top_k
    assert 'blue whale' in aug_result[0].lower()

    # check if logs have been used
    assert logs['augmented']


@pytest.mark.skip_on_ci
def test_rag_openai_gpt(animals_data: list[str], api_key: str) -> None:
    """Test RAG pipeline with OpenAI's GPT."""
    logs = {
        'retrieval': {},
        'augmented': {},
        'generation': {},
    }

    rag = Rago(
        retrieval=StringRet(animals_data),
        augmented=OpenAIAug(api_key=api_key, top_k=3),
        generation=OpenAIGen(api_key=api_key, model_name='gpt-3.5-turbo'),
    )

    query = 'Is there any animal larger than a dinosaur?'
    result = rag.prompt(query)

    assert (
        'blue whale' in result.lower()
    ), 'Expected response to mention Blue Whale as a larger animal.'

    # check if logs have been used
    assert logs['retrieval']
    assert logs['augmented']
    assert logs['generation']
