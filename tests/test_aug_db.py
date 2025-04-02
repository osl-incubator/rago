"""Tests for Rago package: Vector DBs."""

import sys
import tempfile

from functools import partial
from typing import Generator, Optional

import chromadb
import pytest

from chromadb.config import Settings
from rago.augmented.db.chroma import ChromaDB

API_MAP = {
    # ChromaDB: 'api_key_openai',
}

dbs = [
    partial(
        ChromaDB,
        **dict(),
    ),
]


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Provide temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


def create_chroma_client(
    persist_directory: Optional[str] = None,
) -> chromadb.Client:
    """Create a Chroma client instance with specified persist directory."""
    settings = Settings()
    if persist_directory:
        settings = Settings(
            persist_directory=persist_directory, is_persistent=True
        )
    return chromadb.Client(settings=settings)


# def create_chroma_instance(
#     client: chromadb.Client, collection_name: str = 'test_collection'
# ) -> ChromaDB:
#     """Create a Chroma instance with specified client and collection name."""
#     return ChromaDB(client=client, collection_name=collection_name)


@pytest.mark.skipif(
    sys.platform == 'win32',
    reason='Skipping test on Windows due to file locking issues.',
)
@pytest.mark.parametrize(
    'question,expected_answer',
    [
        pytest.param(
            'Is there any animal larger than a dinosaur?', 'Blue Whale', id='0'
        ),
        pytest.param(
            'What animal is renowned as the fastest animal on the planet?',
            'Peregrine Falcon',
            id='1',
        ),
        pytest.param('An animal which do pollination?', 'Honey Bee', id='8'),
    ],
)
@pytest.mark.parametrize('partial_model', dbs)
def test_aug_chromadb(
    request,
    animals_data: list[str],
    question: str,
    expected_answer: str,
    partial_model: partial,
    temp_dir: str,
) -> None:
    """Test RAG pipeline with ChromaDB."""
    question_id = request.node.callspec._idlist[1]
    print(
        f'Running test with ID: {question_id}, \
        question: {question}, \
        expected_answer: {expected_answer}'
    )
    logs = {'augmented': {}}
    top_k = 2
    embedding_size = len(animals_data)  # Choose a fixed embedding size

    # Not needed for this test
    model_class = partial_model.func
    api_key_name: str = API_MAP.get(model_class, '')
    api_key = locals().get(api_key_name, '')

    client = create_chroma_client(temp_dir)

    documents = animals_data
    # Create fixed-size dummy embeddings
    embeddings = [[i] * embedding_size for i in range(len(documents))]

    model_args = {
        'client': client,
        **({'api_key': api_key} if api_key else {}),
    }

    db = partial_model(**model_args)
    db.embed(documents=(documents, embeddings))

    query_encoded = [question_id] * embedding_size
    distances, ids = db.search(query_encoded=query_encoded, top_k=top_k)

    assert len(distances) == 2
    assert len(ids) == 2
    assert ids[0] == question_id
    assert expected_answer.lower() in documents[int(ids[0])].lower()
