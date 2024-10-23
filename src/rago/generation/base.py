"""Base classes for generation."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any


class GenerationBase:
    """Generic Generation class."""

    model: Any
    tokenizer: Any
    output_max_length: int = 500

    @abstractmethod
    def __init__(
        self, model_name: str = '', output_max_length: int = 500
    ) -> None:
        """Initialize GenerationBase."""
        self.model_name = model_name
        self.output_max_length = output_max_length

    @abstractmethod
    def generate(self, query: str, context: list[str]) -> str:
        """Generate text."""
        ...
