"""Base classes for the augmented step."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional

from typeguard import typechecked

from rago.db import DBBase, FaissDB


@typechecked
class AugmentedBase:
    """Define the base structure for Augmented classes."""

    api_key: str = ''
    model: Optional[Any]
    model_name: str = ''
    db: Any
    k: int = 0
    temperature: float = 0.5
    prompt_template: str = ''
    result_separator = '\n'
    output_max_length: int = 500

    # default values to be overwritten by the derived classes
    default_model_name: str = ''
    default_k: int = 0
    default_temperature: float = 0.5
    default_prompt_template: str = (
        'Retrieve {k} entries from the context that better answer the '
        'following query:\n```\n{query}\n```\n\ncontext:\n```\n{context}\n```'
    )
    default_result_separator = '\n'
    default_output_max_length: int = 500

    @abstractmethod
    def __init__(
        self,
        model_name: str = '',
        api_key: str = '',
        db: DBBase = FaissDB(),
        k: int = 0,
        temperature: float = 0.5,
        prompt_template: str = '',
        result_separator: str = '\n',
        output_max_length: int = 500,
    ) -> None:
        """Initialize AugmentedBase."""
        self.db = db
        self.api_key = api_key

        self.k = k or self.default_k
        self.model_name = model_name or self.default_model_name
        self.temperature = temperature or self.default_temperature
        self.result_separator = (
            result_separator or self.default_result_separator
        )
        self.prompt_template = prompt_template or self.default_prompt_template
        self.output_max_length = (
            output_max_length or self.default_output_max_length
        )

        self.model = None

        self._validate()
        self._setup()

    def _validate(self) -> None:
        """Raise an error if the initial parameters are not valid."""
        return

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        return

    @abstractmethod
    def search(
        self,
        query: str,
        documents: Any,
        k: int = 0,
    ) -> list[str]:
        """Search an encoded query into vector database."""
        ...
