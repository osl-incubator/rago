"""Tests for Rago package using OpenAI GPT."""

import os

from typing import cast

import pytest

from rago.generation import OpenAIGen

from ..models import AnimalModel


@pytest.mark.skip_on_ci
def test_rag_openai_gpt_general(
    animals_data: list[str], api_key_openai: str
) -> None:
    """Test RAG pipeline with OpenAI's GPT."""
    logs = {'generation': {}}

    temperature = 0
    expected_answer = 'blue whale'
    context = [
        text for text in animals_data if expected_answer in text.lower()
    ]

    gen = OpenAIGen(
        api_key=api_key_openai,
        model_name='gpt-3.5-turbo',
        logs=logs['generation'],
        temperature=temperature,
    )

    query = 'Is there any animal larger than a dinosaur?'
    result = gen.generate(query, context)

    error_message = (
        f'Expected response: `{expected_answer}`, Result: `{result}`.'
    )

    assert gen.temperature == temperature
    assert expected_answer in result.lower(), error_message
    assert logs['generation']


@pytest.mark.skip_on_ci
@pytest.mark.parametrize(
    'question,expected_answer',
    [
        ('What animal is larger than a dinosaur?', ('Blue Whale',)),
        (
            'What animal is renowned as the fastest animal on the planet?',
            ('Peregrine Falcon', 'Cheetah'),
        ),
    ],
)
def test_rag_openai_gpt_structured_output(
    api_key_openai: str,
    animals_data: list[str],
    question: str,
    expected_answer: tuple[str],
) -> None:
    """Test OpenAI's GPT Generation with structure output."""

    logs = {'generation': {}}

    temperature = 0
    context = [
        text
        for text in animals_data
        if all(
            [expected.lower() in text.lower() for expected in expected_answer]
        )
    ]

    gen = OpenAIGen(
        api_key=api_key_openai,
        model_name='gpt-3.5-turbo',
        logs=logs['generation'],
        temperature=temperature,
        structured_output=AnimalModel,
    )
    assert gen.temperature == temperature
    result = cast(AnimalModel, gen.generate(question, context))

    error_message = (
        f'Expected response to mention `{expected_answer}`. '
        f'Result: `{result.name}`.'
    )

    assert result.name in expected_answer, error_message
    assert logs['generation']
