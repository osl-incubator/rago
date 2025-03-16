"""Tests for Rago package using OpenAI GPT."""

import os
import shutil

from functools import partial
from pathlib import Path

import pytest

from rago import Rago
from rago.augmented import OpenAIAug, SpaCyAug
from rago.extensions.cache import CacheFile
from rago.generation import OpenAIGen
from rago.retrieval import PDFPathRet

PDF_DATA_PATH = Path(__file__).parent / 'data' / 'pdf'
TMP_DIR = Path('/tmp') / 'rago'

RET_CACHE = CacheFile(target_dir=TMP_DIR / 'ret')
AUG_CACHE = CacheFile(target_dir=TMP_DIR / 'aug')
GEN_CACHE = CacheFile(target_dir=TMP_DIR / 'gen')


def clear_folder(folder: Path):
    """
    Remove all files and subdirectories inside the given folder.

    Parameters
    ----------
    folder : Path
        The folder whose contents should be deleted.
    """
    if not folder.exists():
        print(f"Folder '{folder}' does not exist.")
        return

    for item in folder.iterdir():
        if item.is_file():
            item.unlink()  # Remove file
        elif item.is_dir():
            shutil.rmtree(item)  # Remove directory and its contents


def is_directory_empty(directory: Path) -> bool:
    """Check if the directory is not empty."""
    return not os.listdir(directory)


@pytest.fixture
def api_keys(env) -> {str, str}:
    """Fixture for OpenAI API key from environment."""
    keys = {}

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise EnvironmentError(
            'Please set the OPENAI_API_KEY environment variable.'
        )
    keys['OpenAIAug'] = openai_api_key
    keys['OpenAIGen'] = openai_api_key

    return keys


@pytest.mark.skip_on_ci
@pytest.mark.parametrize(
    'aug_class',
    [
        partial(OpenAIAug, top_k=3, cache=AUG_CACHE),
        partial(SpaCyAug, top_k=3, cache=AUG_CACHE),
    ],
)
def test_cache(
    animals_data: list[str], api_keys: dict[str, str], aug_class: partial
) -> None:
    """Test RAG pipeline with OpenAI's GPT."""
    api_name = aug_class.func.__name__
    aug_api_key = api_keys.get(api_name, '')
    gen_api_key = api_keys.get('OpenAIGen', '')

    for cache in [RET_CACHE, AUG_CACHE, GEN_CACHE]:
        clear_folder(cache.target_dir)

    ret = PDFPathRet(PDF_DATA_PATH / '1.pdf', cache=RET_CACHE)
    aug = aug_class(api_key=aug_api_key)
    gen = OpenAIGen(
        api_key=gen_api_key, model_name='gpt-3.5-turbo', cache=GEN_CACHE
    )

    rag = Rago(
        retrieval=ret,
        augmented=aug,
        generation=gen,
    )

    query = 'Is vitamin D effective?'
    rag.prompt(query)

    # note: we don't need to test the gen_cache
    for cache in [RET_CACHE, AUG_CACHE]:
        assert not is_directory_empty(cache.target_dir), (
            f"Cache for {cache} didn't work."
        )
        clear_folder(cache.target_dir)
