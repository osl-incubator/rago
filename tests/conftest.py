"""Configuration for unit tests."""

from __future__ import annotations

import os

from pathlib import Path

import pytest

from dotenv import dotenv_values, load_dotenv


@pytest.fixture
def env() -> dict[str, str]:
    """Return a fixture for the environment variables from .env."""
    dotenv_file = Path(__file__).parent / '.env'
    load_dotenv(dotenv_file)
    return dotenv_values(dotenv_file)


@pytest.fixture
def animals_data() -> list[str]:
    """Fixture for loading the animals dataset."""
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
