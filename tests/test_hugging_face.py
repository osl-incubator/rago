"""Tests for rago package."""

from pathlib import Path

import pytest

from rago import Rago
from rago.augmented import SentenceTransformerAug
from rago.generation import HuggingFaceGen
from rago.retrieval import StringRet


@pytest.fixture
def animals_data() -> list[str]:
    """Create a fixture with animals dataset."""
    data_path = Path(__file__).parent / 'data' / 'animals.txt'
    with open(data_path) as f:
        data = [line.strip() for line in f.readlines() if line.strip()]
        return data


def test_aug_sentence_transformer(animals_data: list[str]) -> None:
    """Test RAG with hugging face."""
    query = 'Is there any animals larger than a dinosaur?'
    top_k = 3

    ret_string = StringRet(animals_data)
    aug_sentence_transformer = SentenceTransformerAug(top_k=top_k)

    ret_result = ret_string.get()
    aug_result = aug_sentence_transformer.search(query, ret_result)

    assert aug_sentence_transformer.top_k == top_k
    assert len(aug_result) == top_k
    assert 'blue whale' in aug_result[0].lower()


def test_rag_hugging_face(animals_data: list[str]) -> None:
    """Test RAG with hugging face."""
    rag = Rago(
        retrieval=StringRet(animals_data),
        augmented=SentenceTransformerAug(top_k=3),
        generation=HuggingFaceGen(),
    )

    query = 'Is there any animals larger than a dinosaur?'
    result = rag.prompt(query)
    assert 'blue whale' in result.lower()
