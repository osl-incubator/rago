"""Tests for rago package."""

import os

import pytest

from rago import Rago
from rago.augmented import SentenceTransformerAug
from rago.generation import LlamaGen
from rago.retrieval import StringRet


@pytest.mark.skip_on_ci
def test_llama(env, animals_data: list[str], device: str = 'auto') -> None:
    """Test RAG with hugging face."""
    HF_TOKEN = os.getenv('HF_TOKEN', '')

    rag = Rago(
        retrieval=StringRet(animals_data),
        augmented=SentenceTransformerAug(top_k=3),
        generation=LlamaGen(api_key=HF_TOKEN, device=device),
    )

    query = 'Is there any animals larger than a dinosaur?'
    result = rag.prompt(query)
    assert 'blue whale' in result.lower()
