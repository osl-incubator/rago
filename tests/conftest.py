"""Configuration for unit tests."""

from __future__ import annotations

import os
import warnings

from pathlib import Path

import pytest

from dotenv import dotenv_values, load_dotenv


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
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError(
            'Please set the OPENAI_API_KEY environment variable.'
        )
    return api_key


@pytest.fixture
def api_key_gemini(env: dict[str, str]) -> str:
    """Fixture for Gemini API key from environment."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise EnvironmentError(
            'Please set the GEMINI_API_KEY environment variable.'
        )
    return api_key


@pytest.fixture
def api_key_hugging_face(env: dict[str, str]) -> str:
    """Fixture for Hugging Face API key from environment."""
    api_key = os.getenv('HF_TOKEN')
    if not api_key:
        raise EnvironmentError('Please set the HF_TOKEN environment variable.')
    return api_key


@pytest.fixture
def api_key_cohere(env) -> str:
    """Fixture for Cohere API key from environment."""
    key = os.getenv('COHERE_API_KEY')
    if not key:
        raise EnvironmentError(
            'Please set the COHERE_API_KEY environment variable.'
        )
    return key


@pytest.fixture
def api_key_fireworks(env) -> str:
    """Fixture for Fireworks API key from environment."""
    key = os.getenv('FIREWORKS_API_KEY')
    if not key:
        raise EnvironmentError(
            'Please set the FIREWORKS_API_KEY environment variable.'
        )
    return key


@pytest.fixture
def api_key_together(env) -> str:
    """Fixture for Together API key from environment."""
    key = os.getenv('TOGETHER_API_KEY')
    if not key:
        raise EnvironmentError(
            'Please set the TOGETHER_API_KEY environment variable.'
        )
    return key
