""""""

from __future__ import annotations

from abc import abstractclassmethod
from typing import Any


class GenerationBase:
    """Generic Generation class."""

    model: Any
    tokenizer: Any
    output_max_length: int = 150

    @abstractclassmethod
    def __init__(
        self, model_name: str = '', output_max_length: int = 150
    ) -> None:
        """Initialize GenerationBase."""
        ...

    @abstractclassmethod
    def run(self, query: str, context: list[str]) -> str:
        """Generate text."""
        ...
