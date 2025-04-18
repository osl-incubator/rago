"""Tests for Rago generation module."""

import platform

from functools import partial
from typing import cast

import pytest

from rago.generation import (
    CohereGen,
    DeepSeekGen,
    FireworksGen,
    GeminiGen,
    GroqGen,
    HuggingFaceGen,
    HuggingFaceInfGen,
    LlamaGen,
    OllamaGen,
    OllamaOpenAIGen,
    OpenAIGen,
    PhiGen,
    TogetherGen,
)

from .models import AnimalModel

# LlamaGen doesn't support temperature zero
TEMPERATURE = 0.0001
GENERATION_LOG = {'generation': {}}

IS_OS_MACOS = platform.system().lower() == 'darwin'

API_MAP = {
    GeminiGen: 'api_key_gemini',
    OpenAIGen: 'api_key_openai',
    HuggingFaceGen: 'api_key_hugging_face',
    HuggingFaceInfGen: 'api_key_hugging_face',
    LlamaGen: 'api_key_hugging_face',
    OllamaGen: '',
    OllamaOpenAIGen: '',
    CohereGen: 'api_key_cohere',
    FireworksGen: 'api_key_fireworks',
    TogetherGen: 'api_key_together',
    GroqGen: 'api_key_groq',
}

gen_models_no_ci = [
    # model 0
    partial(
        OpenAIGen,
        **dict(
            model_name='gpt-3.5-turbo',
        ),
    ),
    # model 1
    partial(
        GeminiGen,
        **dict(
            model_name='gemini-1.5-flash',
        ),
    ),
    # model 2
    partial(
        HuggingFaceGen,
    ),
    # model 3
    partial(
        LlamaGen,
        **dict(device='auto'),
    ),
    # model 4
    partial(
        CohereGen,
        **dict(
            model_name='command-r-plus-08-2024',
        ),
    ),
    # model 5
    partial(
        DeepSeekGen,
        **dict(
            device='auto',
        ),
    ),
    # model 6
    partial(
        FireworksGen,
    ),
    # model 7
    partial(
        TogetherGen,
        **dict(model_name='meta-llama/Llama-3.3-70B-Instruct-Turbo-Free'),
    ),
    # model 8
    partial(
        HuggingFaceInfGen,
    ),
    # model 9
    partial(
        GroqGen,
    ),
    # model 10
    partial(
        PhiGen,
        **dict(model_name='microsoft/phi-2'),
    ),
]


gen_models_ci = [
    # model 0
    partial(
        OllamaOpenAIGen,
    ),
    # model 1
    partial(
        OllamaGen,
    ),
]


def _generation_simple_output(
    animals_data: list[str],
    api_key_openai: str,
    api_key_cohere: str,
    api_key_fireworks: str,
    api_key_gemini: str,
    api_key_together: str,
    api_key_hugging_face: str,
    api_key_groq: str,
    partial_model: partial,
) -> bool:
    """Test RAG pipeline with model generation."""
    model_class = partial_model.func

    if IS_OS_MACOS and issubclass(
        model_class,
        (
            OllamaGen,
            OllamaOpenAIGen,
        ),
    ):
        pytest.skip(f'Skipping {model_class} on macOS due to CI failure.')

    api_key_name: str = API_MAP.get(model_class, '')
    api_key = locals().get(api_key_name, '')

    model_args = {
        'temperature': TEMPERATURE,
        'logs': GENERATION_LOG['generation'],
        **({'api_key': api_key} if api_key else {}),
    }

    expected_answer = 'peregrine falcon'
    context = [
        text for text in animals_data if expected_answer in text.lower()
    ]

    gen_model = partial_model(**model_args)

    query = 'what is the fastest bird on the earth?'
    result = gen_model.generate(query, context)

    error_message = (
        f'Expected response: `{expected_answer}`, Result: `{result}`.'
    )

    assert gen_model.temperature == TEMPERATURE
    assert expected_answer.lower() in result.lower(), error_message
    assert GENERATION_LOG['generation']

    return True


@pytest.mark.skip_on_ci
@pytest.mark.parametrize('partial_model', gen_models_no_ci)
def test_generation_simple_output(
    animals_data: list[str],
    api_key_openai: str,
    api_key_cohere: str,
    api_key_fireworks: str,
    api_key_gemini: str,
    api_key_together: str,
    api_key_hugging_face: str,
    api_key_groq: str,
    partial_model: partial,
) -> None:
    """Test RAG pipeline with model generation."""
    assert _generation_simple_output(
        animals_data,
        api_key_openai,
        api_key_cohere,
        api_key_fireworks,
        api_key_gemini,
        api_key_together,
        api_key_hugging_face,
        api_key_groq,
        partial_model,
    )


@pytest.mark.parametrize('partial_model', gen_models_ci)
def test_generation_simple_output(
    animals_data: list[str],
    api_key_openai: str,
    api_key_cohere: str,
    api_key_fireworks: str,
    api_key_gemini: str,
    api_key_together: str,
    api_key_hugging_face: str,
    api_key_groq: str,
    partial_model: partial,
) -> None:
    """Test RAG pipeline with model generation."""
    assert _generation_simple_output(
        animals_data,
        api_key_openai,
        api_key_cohere,
        api_key_fireworks,
        api_key_gemini,
        api_key_together,
        api_key_hugging_face,
        api_key_groq,
        partial_model,
    )


def _generation_structure_output(
    api_key_openai: str,
    api_key_cohere: str,
    api_key_fireworks: str,
    api_key_gemini: str,
    api_key_together: str,
    api_key_hugging_face: str,
    api_key_groq: str,
    animals_data: list[str],
    question: str,
    partial_model: partial,
    expected_answer: tuple[str],
) -> bool:
    """Test Model Generation with structure output."""
    model_class = partial_model.func

    # Skip structured output for models that don't support it
    if issubclass(
        model_class,
        (
            DeepSeekGen,
            HuggingFaceInfGen,
            HuggingFaceGen,
            LlamaGen,
            OllamaGen,
        ),
    ):
        pytest.skip(f"{model_class} doesn't support structured output.")

    if IS_OS_MACOS and issubclass(
        model_class,
        (
            OllamaGen,
            OllamaOpenAIGen,
        ),
    ):
        pytest.skip(f'Skipping {model_class} on macOS due to CI failure.')

    api_key_name: str = API_MAP.get(model_class, '')
    api_key = locals().get(api_key_name, '')

    model_args = {
        'temperature': TEMPERATURE,
        'logs': GENERATION_LOG['generation'],
        'structured_output': AnimalModel,
        **({'api_key': api_key} if api_key else {}),
    }

    gen_model = partial_model(**model_args)

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

    assert result.name.lower() in expected_answer, error_message
    assert GENERATION_LOG['generation']
    return True


@pytest.mark.skip_on_ci
@pytest.mark.parametrize(
    'question,expected_answer',
    [
        ('what is the fastest bird on the earth?', ('peregrine falcon',)),
    ],
)
@pytest.mark.parametrize('partial_model', gen_models_no_ci)
def test_generation_structure_output(
    api_key_openai: str,
    api_key_cohere: str,
    api_key_fireworks: str,
    api_key_gemini: str,
    api_key_together: str,
    api_key_hugging_face: str,
    api_key_groq: str,
    animals_data: list[str],
    question: str,
    partial_model: partial,
    expected_answer: tuple[str],
) -> None:
    """Test Model Generation with structure output."""
    assert _generation_structure_output(
        api_key_openai,
        api_key_cohere,
        api_key_fireworks,
        api_key_gemini,
        api_key_together,
        api_key_hugging_face,
        api_key_groq,
        animals_data,
        question,
        partial_model,
        expected_answer,
    )


@pytest.mark.parametrize(
    'question,expected_answer',
    [
        ('what is the fastest bird on the earth?', ('peregrine falcon',)),
    ],
)
@pytest.mark.parametrize('partial_model', gen_models_ci)
def test_generation_structure_output(
    api_key_openai: str,
    api_key_cohere: str,
    api_key_fireworks: str,
    api_key_gemini: str,
    api_key_together: str,
    api_key_hugging_face: str,
    api_key_groq: str,
    animals_data: list[str],
    question: str,
    partial_model: partial,
    expected_answer: tuple[str],
) -> None:
    """Test Model Generation with structure output."""
    assert _generation_structure_output(
        api_key_openai,
        api_key_cohere,
        api_key_fireworks,
        api_key_gemini,
        api_key_together,
        api_key_hugging_face,
        api_key_groq,
        animals_data,
        question,
        partial_model,
        expected_answer,
    )
