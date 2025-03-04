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
def test_rag_openai_gpt_general(animals_data: list[str], api_key: str) -> None:
    """Test RAG pipeline with OpenAI's GPT."""
    logs = {
        'retrieval': {},
        'augmented': {},
        'generation': {},
    }

    temperature = 0

    ret = StringRet(animals_data, logs=logs['retrieval'])
    gen = OpenAIGen(
        api_key=api_key,
        model_name='gpt-3.5-turbo',
        logs=logs['generation'],
        temperature=temperature,
    )
    aug = OpenAIAug(api_key=api_key, top_k=3, logs=logs['augmented'])

    assert gen.temperature == temperature

    rag = Rago(
        retrieval=ret,
        augmented=aug,
        generation=gen,
    )

    query = 'Is there any animal larger than a dinosaur?'
    result = rag.prompt(query)

    assert 'blue whale' in result.lower(), (
        'Expected response to mention Blue Whale as a larger animal.'
    )

    # check if logs have been used
    assert logs['retrieval']
    assert logs['augmented']
    assert logs['generation']
