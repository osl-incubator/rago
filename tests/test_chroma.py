"""Test ChromaDB implementation."""

import tempfile

from typing import Generator

import chromadb
import pytest

from chromadb.config import Settings
from rago.augmented.db.chroma import ChromaDB


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Provide temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


def create_chroma_client(
    persist_directory: str | None = None,
) -> chromadb.Client:
    """Create a Chroma client instance with specified persist directory."""
    settings = Settings()
    if persist_directory:
        settings = Settings(
            persist_directory=persist_directory, is_persistent=True
        )
    return chromadb.Client(settings=settings)


def create_chroma_instance(
    client: chromadb.Client, collection_name: str = 'test_collection'
) -> ChromaDB:
    """Create a Chroma instance with specified client and collection name."""
    return ChromaDB(client=client, collection_name=collection_name)


def test_chroma_basic(temp_dir: str) -> None:
    """Test basic ChromaDB functionality with persistence."""
    client = create_chroma_client(temp_dir)
    db = create_chroma_instance(client)
    documents = ['doc1', 'doc2', 'doc3']
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    db.embed(documents=documents, embeddings=embeddings)
    distances, ids = db.search(query_encoded=[1.0, 0.0, 0.0], top_k=1)
    assert len(distances) == 1
    assert len(ids) == 1
    assert ids[0] == '0'


def test_chroma_no_persist() -> None:
    """Test basic ChromaDB functionality without persistence."""
    client = create_chroma_client()
    db = create_chroma_instance(client, collection_name='test_no_persist')
    documents = ['doc1', 'doc2', 'doc3']
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    db.embed(documents=documents, embeddings=embeddings)
    distances, ids = db.search(query_encoded=[1.0, 0.0, 0.0], top_k=1)
    assert len(distances) == 1
    assert len(ids) == 1
    assert ids[0] == '0'
