"""Tests for rago package."""

from pathlib import Path

import pytest

from rago import Rago
from rago.augmented import HuggingFaceAug
from rago.generation import LlamaGen
from rago.retrieval import StringRet


@pytest.fixture
def animals_data() -> list[str]:
    """Create a fixture with animals dataset."""
    data_path = Path(__file__).parent / 'data' / 'animals.txt'
    with open(data_path) as f:
        data = [line.strip() for line in f.readlines() if line.strip()]
        return data


@pytest.mark.skip_on_ci
def test_llama(env, animals_data: list[str], device: str = 'auto') -> None:
    """Test RAG with hugging face."""
    HF_TOKEN = env.get('HF_TOKEN', '')

    rag = Rago(
        retrieval=StringRet(animals_data),
        augmented=HuggingFaceAug(k=3),
        generation=LlamaGen(apikey=HF_TOKEN, device=device),
    )

    query = 'Is there any animals larger than a dinosaur?'
    result = rag.prompt(query)
    assert 'Blue Whale' in result