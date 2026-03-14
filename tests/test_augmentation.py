"""Tests for Rago package for augmentation."""

from functools import partial

import pytest

from rago.augmented import Augmented

from tests.helpers import call_or_skip, partial_backend, require_spacy_model

CohereAug = partial(
    Augmented, backend='cohere', model_name='embed-english-light-v3.0'
)
FireworksAug = partial(
    Augmented,
    backend='fireworks',
    model_name='accounts/fireworks/models/qwen3-embedding-8b',
)
OpenAIAug = partial(
    Augmented, backend='openai', model_name='text-embedding-3-small'
)
SpaCyAug = partial(Augmented, backend='spacy', model_name='en_core_web_md')
TogetherAug = partial(
    Augmented, backend='together', model_name='BAAI/bge-base-en-v1.5'
)

API_MAP = {
    'cohere': 'api_key_cohere',
    'fireworks': 'api_key_fireworks',
    'openai': 'api_key_openai',
    'together': 'api_key_together',
}

gen_models = [
    # model 0
    SpaCyAug,
    # model 1
    OpenAIAug,
    # model 2
    CohereAug,
    # model 3
    FireworksAug,
    # model 4
    TogetherAug,
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
    top_k = 2

    backend = partial_backend(partial_model)
    api_key_name: str = API_MAP.get(backend, '')
    api_key = locals().get(api_key_name, '')

    model_args = {
        'top_k': top_k,
        **({'api_key': api_key} if api_key else {}),
    }

    if backend == 'spacy':
        require_spacy_model('en_core_web_md')

    gen_model = call_or_skip(
        backend or 'augmented', partial_model, **model_args
    )
    aug_result = call_or_skip(
        backend or 'augmented',
        gen_model.search,
        question,
        animals_data,
    )
    assert gen_model.params.top_k == top_k
    assert top_k >= len(aug_result)
    assert any(
        expected_answer.lower() in result.lower() for result in aug_result
    )
