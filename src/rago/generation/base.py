"""Base classes for generation."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from typeguard import typechecked


@typechecked
class GenerationBase:
    """Generic Generation class."""

    model: Any
    tokenizer: Any
    output_max_length: int = 500

    @abstractmethod
    def __init__(
        self, model_name: str = '', output_max_length: int = 500
    ) -> None:
        """Initialize GenerationBase.

        Parameters
        ----------
        model_name : str
            The name of the model to use.
        output_max_length : int
            Maximum length of the generated output.
        """
        self.model_name = model_name
        self.output_max_length = output_max_length

    @abstractmethod
    def generate(
        self, query: str, context: list[str], language: str = 'en'
    ) -> str:
        """Generate text with optional language parameter.

        Parameters
        ----------
        query : str
            The input query or prompt.
        context : list[str]
            Additional context information for the generation.
        language : str, optional
            The language for generation, by default 'en'.

        Returns
        -------
        str
            Generated text based on query and context.
        """
        ...
