"""Base classes for the augmented step."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from torch import Tensor
from typeguard import typechecked
from typing_extensions import TypeAlias

from rago.augmented.db import DBBase, FaissDB
from rago.base import RagoBase
from rago.extensions.cache import Cache

EmbeddingType: TypeAlias = Union[
    npt.NDArray[np.float64],
    Tensor,
    list[Tensor],
]


@typechecked
class AugmentedBase(RagoBase):
    """Define the base structure for Augmented classes."""

    model: Optional[Any]
    model_name: str = ''
    db: Any
    top_k: int = 0

    # default values to be overwritten by the derived classes
    default_model_name: str = ''
    default_top_k: int = 0

    def __init__(
        self,
        model_name: str = '',
        db: DBBase = FaissDB(),
        top_k: int = 0,
        api_key: str = '',
        cache: Optional[Cache] = None,
        logs: dict[str, Any] = {},
    ) -> None:
        """Initialize AugmentedBase."""
        super().__init__(api_key=api_key, cache=cache, logs=logs)

        self.db = db

        self.top_k = top_k or self.default_top_k
        self.model_name = model_name or self.default_model_name
        self.model = None

        self._validate()
        self._setup()

    def _validate(self) -> None:
        """Raise an error if the initial parameters are not valid."""
        return

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        return

    def get_embedding(self, content: list[str]) -> EmbeddingType:
        """Retrieve the embedding for a given text using OpenAI API."""
        raise Exception('Method not implemented.')

    @abstractmethod
    def search(
        self,
        query: str,
        documents: Any,
        top_k: int = 0,
    ) -> list[str]:
        """Search an encoded query into vector database."""
        ...
