"""Tests for rago package."""

from rago import Rago
from rago.augmented import SentenceTransformerAug
from rago.generation import HuggingFaceGen
from rago.retrieval import StringRet


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
