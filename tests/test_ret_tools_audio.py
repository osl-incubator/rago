"""Test audio retrieval."""

from pathlib import Path
from unittest.mock import patch

import pytest

from rago.retrieval import AudioPathRet, Retrieval

AUDIO_DATA_PATH = Path(__file__).parent / 'data' / 'audio'


def test_audio_retrieval_transcription() -> None:
    """Test audio transcription retrieval."""
    audio_ret = AudioPathRet(AUDIO_DATA_PATH / 'test.wav')

    with patch(
        'rago.retrieval.file.transcribe_audio',
        return_value='Hello from audio',
    ):
        chunks = audio_ret.get()

    assert chunks
    assert 'Hello from audio' in chunks[0]


def test_audio_retrieval_rejects_unsupported_file() -> None:
    """Test that unsupported audio formats are rejected."""
    unsupported_file = AUDIO_DATA_PATH / 'unsupported.txt'
    unsupported_file.write_text('not an audio file')

    try:
        with pytest.raises(Exception, match='supported audio format'):
            AudioPathRet(unsupported_file)
    finally:
        unsupported_file.unlink()


def test_audio_retrieval_missing_file() -> None:
    """Test that a missing audio file raises an error."""
    with pytest.raises(Exception, match="File doesn't exist"):
        AudioPathRet(AUDIO_DATA_PATH / 'missing.wav')


def test_audio_retrieval_public_api() -> None:
    """Test audio retrieval through the public Retrieval API."""
    audio_ret = Retrieval(
        source=AUDIO_DATA_PATH / 'test.wav',
        backend='audio',
    )

    with patch(
        'rago.retrieval.file.transcribe_audio',
        return_value='Hello from public audio API',
    ):
        chunks = audio_ret.get()

    assert chunks
    assert 'Hello from public audio API' in chunks[0]


def test_audio_retrieval_public_api() -> None:
    """Test audio retrieval through the public Retrieval API."""
    audio_ret = Retrieval(
        backend='audio',
        source=AUDIO_DATA_PATH / 'test.wav',
    )

    with patch(
        'rago.retrieval.file.transcribe_audio',
        return_value='Hello from audio',
    ):
        chunks = audio_ret.get()

    assert chunks
    assert 'Hello from audio' in chunks[0]
