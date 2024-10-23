"""Configuration for unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from dotenv import dotenv_values, load_dotenv


@pytest.fixture
def env() -> dict[str, str]:
    """Return a fixture for the environment variables from .env."""
    dotenv_file = Path(__file__).parent / '.env'
    load_dotenv(dotenv_file)
    return dotenv_values(dotenv_file)
