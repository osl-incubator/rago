"""Tests for Rago package for augmentation."""

from functools import partial

import pytest

from rago.augmented import (
    CohereAug,
    FireworksAug,
    OpenAIAug,
    SpaCyAug,
    TogetherAug,
)

API_MAP = {
    OpenAIAug: 'api_key_openai',
    CohereAug: 'api_key_cohere',
    FireworksAug: 'api_key_fireworks',
    TogetherAug: 'api_key_together',
}

gen_models = [
    # model 0
    partial(
        SpaCyAug,
        **dict(
            model_name='en_core_web_md',
        ),
    ),
    # model 1
    partial(
        OpenAIAug,
        **dict(
            model_name='text-embedding-3-small',
        ),
    ),
    # model 2
    partial(
        CohereAug,
    ),
    # model 3
    partial(
        FireworksAug,
    ),
    # model 4
    partial(TogetherAug, **dict(model_name='BAAI/bge-base-en-v1.5')),
]


@pytest.mark.skip_on_ci
@pytest.mark.parametrize(
    'question,expected_answer',
    [
        ('Is there any animal larger than a dinosaur?', 'Blue Whale'),
        (
            'What animal is renowned as the fastest animal on the planet?',
            'Peregrine Falcon',
        ),
        ('An animal which do pollination?', 'Honey Bee'),
    ],
)
@pytest.mark.parametrize('partial_model', gen_models)
def test_aug_spacy(
    animals_data: list[str],
    question: str,
    expected_answer: str,
    api_key_openai: str,
    api_key_cohere: str,
    api_key_gemini: str,
    api_key_fireworks: str,
    api_key_together: str,
    api_key_hugging_face: str,
    partial_model: partial,
) -> None:
    """Test RAG pipeline with SpaCy."""
    logs = {'augmented': {}}
    top_k = 2

    model_class = partial_model.func

    api_key_name: str = API_MAP.get(model_class, '')
    api_key = locals().get(api_key_name, '')

    model_args = {
        'top_k': top_k,
        'logs': logs['augmented'],
        **({'api_key': api_key} if api_key else {}),
    }

    gen_model = partial_model(**model_args)

    aug_result = gen_model.search(question, animals_data)
    assert gen_model.top_k == top_k
    assert top_k >= len(aug_result)
    assert any(
        [expected_answer.lower() in result.lower() for result in aug_result]
    )

    # check if logs have been used
    assert logs['augmented']
