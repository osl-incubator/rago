"""Tests for Rago package using OpenAI GPT."""

import os
import shutil

from functools import partial
from pathlib import Path

import pytest

from rago import Rago
from rago.augmented import Augmented
from rago.extensions.cache import CacheFile
from rago.generation import Generation
from rago.retrieval import Retrieval

from tests.helpers import (
    get_api_key_fixture,
    partial_backend,
    require_spacy_model,
    skip_if_runtime_unavailable,
)

PDF_DATA_PATH = Path(__file__).parent / 'data' / 'pdf'
TMP_DIR = Path('/tmp') / 'rago'

RET_CACHE = CacheFile(target_dir=TMP_DIR / 'ret')
AUG_CACHE = CacheFile(target_dir=TMP_DIR / 'aug')
GEN_CACHE = CacheFile(target_dir=TMP_DIR / 'gen')

AUG_API_MAP = {'openai': 'api_key_openai'}

OpenAIAug = partial(
    Augmented, backend='openai', model_name='text-embedding-3-small'
)
SpaCyAug = partial(Augmented, backend='spacy', model_name='en_core_web_md')
PDFPathRet = partial(Retrieval, backend='pdf')

OpenAIGen = partial(Generation, backend='openai')


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


@pytest.mark.skip_on_ci
@pytest.mark.parametrize(
    'aug_class',
    [
        partial(OpenAIAug, top_k=3),
        partial(SpaCyAug, top_k=3),
    ],
)
def test_cache(
    request: pytest.FixtureRequest,
    animals_data: list[str],
    aug_class: partial,
) -> None:
    """Test RAG pipeline with OpenAI's GPT."""
    backend = partial_backend(aug_class)
    aug_api_key = get_api_key_fixture(request, AUG_API_MAP.get(backend, ''))
    gen_api_key = get_api_key_fixture(request, 'api_key_openai')

    for cache in [RET_CACHE, AUG_CACHE, GEN_CACHE]:
        clear_folder(cache.target_dir)

    if backend == 'spacy':
        require_spacy_model('en_core_web_md')

    try:
        ret = PDFPathRet(PDF_DATA_PATH / '1.pdf', cache=RET_CACHE)
        aug = aug_class(api_key=aug_api_key, cache=AUG_CACHE)
        gen = OpenAIGen(
            api_key=gen_api_key,
            model_name='gpt-3.5-turbo',
            cache=GEN_CACHE,
        )
        rag = Rago() | ret | aug | gen

        query = 'Is vitamin D effective?'
        rag.run(query=query, source=animals_data)
    except Exception as exc:
        skip_if_runtime_unavailable(f'{backend}/openai', exc)
        raise

    # note: we don't need to test the gen_cache
    for cache in [RET_CACHE, AUG_CACHE]:
        assert not is_directory_empty(cache.target_dir), (
            f"Cache for {cache} didn't work."
        )
        clear_folder(cache.target_dir)
