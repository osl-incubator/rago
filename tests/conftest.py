"""Configuration for unit tests."""

from __future__ import annotations

import os
import warnings

from pathlib import Path

import pytest

from dotenv import dotenv_values, load_dotenv

INVALID_API_KEYS = frozenset({'', 'nokey'})


def _load_api_key(variable_name: str) -> str:
    """Return a real API key or skip the provider-backed test."""
    api_key = os.getenv(variable_name, '').strip()
    if api_key.lower() in INVALID_API_KEYS:
        pytest.skip(f'{variable_name} is not configured for this test run.')
    return api_key


@pytest.fixture
def env() -> dict[str, str]:
    """Return a fixture for the environment variables from .env."""
    dotenv_file = Path(__file__).parent / '.env'
    if not dotenv_file.exists():
        warnings.warn('No .env file found.')
        return {}
    load_dotenv(dotenv_file)
    return dotenv_values(dotenv_file)


@pytest.fixture
def animals_data() -> list[str]:
    """Fixture for loading the "animals" dataset."""
    data_path = Path(__file__).parent / 'data' / 'animals.txt'
    with open(data_path) as f:
        data = [line.strip() for line in f.readlines() if line.strip()]
        return data


@pytest.fixture
def api_key_openai(env: dict[str, str]) -> str:
    """Fixture for OpenAI API key from environment."""
    del env
    return _load_api_key('OPENAI_API_KEY')


@pytest.fixture
def api_key_gemini(env: dict[str, str]) -> str:
    """Fixture for Gemini API key from environment."""
    del env
    return _load_api_key('GEMINI_API_KEY')


@pytest.fixture
def api_key_hugging_face(env: dict[str, str]) -> str:
    """Fixture for Hugging Face API key from environment."""
    del env
    return _load_api_key('HF_TOKEN')


@pytest.fixture
def api_key_cohere(env: dict[str, str]) -> str:
    """Fixture for Cohere API key from environment."""
    del env
    return _load_api_key('COHERE_API_KEY')


@pytest.fixture
def api_key_fireworks(env: dict[str, str]) -> str:
    """Fixture for Fireworks API key from environment."""
    del env
    return _load_api_key('FIREWORKS_API_KEY')


@pytest.fixture
def api_key_together(env: dict[str, str]) -> str:
    """Fixture for Together API key from environment."""
    del env
    return _load_api_key('TOGETHER_API_KEY')


@pytest.fixture
def api_key_groq(env: dict[str, str]) -> str:
    """Fixture for GROQ API key from environment."""
    del env
    return _load_api_key('GROQ_API_KEY')
