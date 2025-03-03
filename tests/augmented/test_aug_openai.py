"""Tests for Rago package using OpenAI GPT."""

import os

from typing import cast

import pytest

from rago import Rago
from rago.augmented import OpenAIAug
from rago.generation import OpenAIGen
from rago.retrieval import StringRet

from ..models import AnimalModel


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
def test_aug_openai(
    animals_data: list[str],
    api_key_openai: str,
    question: str,
    expected_answer: str,
) -> None:
    """Test RAG pipeline with OpenAI's GPT."""
    logs = {
        'augmented': {},
    }

    top_k = 2

    ret_string = StringRet(animals_data)
    aug_openai = OpenAIAug(
        api_key=api_key_openai,
        top_k=top_k,
        logs=logs['augmented'],
    )

    ret_result = ret_string.get()
    aug_result = aug_openai.search(question, ret_result)

    assert aug_openai.top_k == top_k
    # note: openai as augmented doesn't work as expected
    #   it is returning a very poor result
    #   it needs to be revisited and improved
    assert len(aug_result) >= 1
    assert expected_answer.lower() in aug_result[0].lower()

    # check if logs have been used
    assert logs['augmented']
