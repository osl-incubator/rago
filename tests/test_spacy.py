"""Tests for Rago package using SpaCy."""

import pytest

from rago.augmented import SpaCyAug


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
def test_aug_spacy(
    animals_data: list[str], question: str, expected_answer: str
) -> None:
    """Test RAG pipeline with SpaCy."""
    logs = {
        'augmented': {},
    }

    top_k = 2

    aug_openai = SpaCyAug(
        model_name='en_core_web_md',
        top_k=top_k,
        logs=logs['augmented'],
    )

    aug_result = aug_openai.search(question, animals_data)

    assert aug_openai.top_k == top_k
    assert top_k >= len(aug_result)
    assert any(
        [expected_answer.lower() in result.lower() for result in aug_result]
    )

    # check if logs have been used
    assert logs['augmented']
