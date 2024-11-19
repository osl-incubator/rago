"""Tests for Rago package using SpaCy."""

from rago.augmented import SpaCyAug
from rago.retrieval import StringRet


def test_aug_spacy(animals_data: list[str]) -> None:
    """Test RAG pipeline with SpaCy."""
    logs = {
        'augmented': {},
    }

    query = 'Is there any animal larger than a dinosaur?'
    top_k = 3

    ret_string = StringRet(animals_data)
    aug_openai = SpaCyAug(
        top_k=top_k,
        logs=logs['augmented'],
    )

    ret_result = ret_string.get()
    aug_result = aug_openai.search(query, ret_result)

    assert aug_openai.top_k == top_k
    assert len(aug_result) == top_k
    assert any(['blue whale' in result.lower() for result in aug_result])

    # check if logs have been used
    assert logs['augmented']
