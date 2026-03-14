"""Regression tests for shared log dictionary handling."""

from __future__ import annotations

from typing import Any

from rago.extensions.logs import Logs
from rago.generation import Generation
from rago.generation.base import GenerationBase


class DummyGeneration(GenerationBase):
    """Minimal generator used to verify shared log state."""

    default_model_name = 'dummy'

    def _load_optional_modules(self) -> None:
        """Avoid loading optional runtime dependencies in tests."""

    def _setup(self) -> None:
        """Skip backend setup for the dummy generator."""

    def generate(self, query: str, data: list[str]) -> str:
        """Return a deterministic response for test assertions."""
        del query, data
        return 'ok'


def test_logs_helper_preserves_explicit_empty_dict() -> None:
    """Keep the caller-provided empty dict instead of replacing it."""
    target: dict[str, Any] = {}

    config = Logs(target)

    assert config.logs is target


def test_generation_base_preserves_explicit_empty_logs_dict() -> None:
    """Mutate the exact log dict passed to a concrete generator."""
    shared_logs: dict[str, Any] = {}

    generator = DummyGeneration(logs=shared_logs)
    result = generator.generate('question', ['context'])

    assert result == 'ok'
    assert generator.logs is shared_logs
    assert shared_logs['result'] == 'ok'


def test_generation_wrapper_preserves_explicit_empty_logs_dict(
    monkeypatch: Any,
) -> None:
    """Forward an explicit empty log dict through lazy resolution."""
    shared_logs: dict[str, Any] = {}

    monkeypatch.setattr(
        'rago.generation.llama.OllamaGen',
        DummyGeneration,
    )

    generator = Generation(backend='ollama', logs=shared_logs)
    result = generator.generate('question', ['context'])

    assert result == 'ok'
    assert generator.logs is shared_logs
    assert shared_logs['result'] == 'ok'
