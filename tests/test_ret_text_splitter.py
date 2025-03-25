"""Tests for text splitter."""

from __future__ import annotations

from rago.retrieval.text_splitter import LangChainTextSplitter


def test_langchain_text_splitter_with_separator(
    animals_data: list[str],
) -> None:
    """Test the langchain text splitter."""
    text = '/n'.join(animals_data)

    # /n can also be used as separator with langchain splitter
    max_chunk_size_original = max([len(line) for line in animals_data])

    # chunk_size is the max size of the chunk
    chunk_size = max_chunk_size_original
    chunk_overlap = 100
    splitter = LangChainTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    result = splitter.split(text)

    for chunk in result:
        assert len(chunk) < chunk_size

    assert len(result) >= len(animals_data)


def test_langchain_text_splitter_without_separator(
    animals_data: list[str],
) -> None:
    """Test the langchain text splitter."""
    text = '. '.join(animals_data)

    # chunk_size is the max size of the chunk
    chunk_size = len(text) // len(animals_data)
    chunk_overlap = 100
    splitter = LangChainTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    result = splitter.split(text)

    for chunk in result:
        assert len(chunk) < chunk_size

    assert len(result) >= len(animals_data)
