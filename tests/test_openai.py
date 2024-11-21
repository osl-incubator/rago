"""Tests for Rago package using OpenAI GPT."""

import os

from typing import cast

import pytest

from rago import Rago
from rago.augmented import OpenAIAug
from rago.generation import OpenAIGen
from rago.retrieval import StringRet

from .models import AnimalModel


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
    # note: openai as augmented doesn't work as expected
    #   it is returning a very poor result
    #   it needs to be revisited and improved
    assert len(aug_result) >= 1
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
        retrieval=StringRet(animals_data, logs=logs['retrieval']),
        augmented=OpenAIAug(api_key=api_key, top_k=3, logs=logs['augmented']),
        generation=OpenAIGen(
            api_key=api_key,
            model_name='gpt-3.5-turbo',
            logs=logs['generation'],
        ),
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


@pytest.mark.skip_on_ci
@pytest.mark.parametrize(
    'question,expected_answer',
    [
        ('What animal is larger than a dinosaur?', 'Blue Whale'),
        (
            'What animal is renowned as the fastest animal on the planet?',
            'Peregrine Falcon',
        ),
    ],
)
def test_rag_openai_gpt_structured_output(
    api_key: str,
    animals_data: list[str],
    question: str,
    expected_answer: str,
) -> None:
    """Test RAG pipeline with OpenAI's GPT."""
    logs = {
        'retrieval': {},
        'augmented': {},
        'generation': {},
    }

    rag = Rago(
        retrieval=StringRet(animals_data, logs=logs['retrieval']),
        augmented=OpenAIAug(api_key=api_key, top_k=3, logs=logs['augmented']),
        generation=OpenAIGen(
            api_key=api_key,
            model_name='gpt-3.5-turbo',
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
