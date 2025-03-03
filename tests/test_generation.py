"""Tests for Rago package using OpenAI GPT."""

import os

from functools import partial
from typing import cast

import pytest

from rago.generation import (
    GeminiGen,
    GenerationBase,
    HuggingFaceGen,
    LlamaGen,
    OpenAIGen,
)

from .models import AnimalModel

TEMPERATURE = 0
GENERATION_LOG = {'generation': {}}

API_MAP = {
    GeminiGen: 'api_key_gemini',
    OpenAIGen: 'api_key_openai',
    HuggingFaceGen: 'api_key_hugging_face',
    LlamaGen: 'api_key_hugging_face',
}

gen_models = [
    partial(
        OpenAIGen,
        **dict(
            model_name='gpt-3.5-turbo',
            temperature=TEMPERATURE,
            logs=GENERATION_LOG['generation'],
        ),
    ),
    partial(
        GeminiGen,
        **dict(
            model_name='gemini-1.5-flash',
            temperature=TEMPERATURE,
            logs=GENERATION_LOG['generation'],
        ),
    ),
    partial(
        HuggingFaceGen,
        **dict(
            temperature=TEMPERATURE,
            logs=GENERATION_LOG['generation'],
        ),
    ),
    partial(
        LlamaGen,
        **dict(
            device='auto',
            temperature=TEMPERATURE,
            logs=GENERATION_LOG['generation'],
        ),
    ),
]


@pytest.mark.skip_on_ci
@pytest.mark.parametrize('partial_model', gen_models)
def test_rag_openai_gpt_general(
    animals_data: list[str],
    api_key_openai: str,
    api_key_gemini: str,
    api_key_hugging_face: str,
    partial_model: partial,
) -> None:
    """Test RAG pipeline with OpenAI's GPT."""
    api_key_name: str = API_MAP.get(partial_model.func, '')
    api_key = locals().get(api_key_name, '')

    expected_answer = 'blue whale'
    context = [
        text for text in animals_data if expected_answer in text.lower()
    ]

    gen_model = partial_model(**{'api_key': key for key in [api_key] if key})

    query = 'Is there any animal larger than a dinosaur?'
    result = gen_model.generate(query, context)

    error_message = (
        f'Expected response: `{expected_answer}`, Result: `{result}`.'
    )

    assert gen_model.temperature == TEMPERATURE
    assert expected_answer in result.lower(), error_message
    assert GENERATION_LOG['generation']


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
@pytest.mark.parametrize('partial_model', gen_models)
def test_rag_openai_gpt_structured_output(
    api_key_openai: str,
    api_key_gemini: str,
    api_key_hugging_face: str,
    animals_data: list[str],
    question: str,
    partial_model: partial,
    expected_answer: tuple[str],
) -> None:
    """Test OpenAI's GPT Generation with structure output."""
    api_key_name: str = API_MAP.get(partial_model.func, '')
    api_key = locals().get(api_key_name, '')

    gen_model = partial_model(
        structured_output=AnimalModel,
        **{'api_key': key for key in [api_key] if key},
    )

    context = [
        text
        for text in animals_data
        if all(
            [expected.lower() in text.lower() for expected in expected_answer]
        )
    ]

    assert gen_model.temperature == TEMPERATURE
    result = cast(AnimalModel, gen_model.generate(question, context))

    error_message = (
        f'Expected response to mention `{expected_answer}`. '
        f'Result: `{result.name}`.'
    )

    assert result.name in expected_answer, error_message
    assert GENERATION_LOG['generation']
