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
    logs = {
        'retrieval': {},
        'augmented': {},
        'generation': {},
    }

    HF_TOKEN = os.getenv('HF_TOKEN', '')

    rag = Rago(
        retrieval=StringRet(
            animals_data,
            logs=logs['retrieval'],
        ),
        augmented=SentenceTransformerAug(
            top_k=3,
            logs=logs['augmented'],
        ),
        generation=LlamaGen(
            api_key=HF_TOKEN,
            device=device,
            logs=logs['generation'],
        ),
    )

    query = 'Is there any animals larger than a dinosaur?'
    result = rag.prompt(query)
    assert 'blue whale' in result.lower()

    # check if logs have been used
    assert logs['retrieval']
    assert logs['augmented']
    assert logs['generation']
