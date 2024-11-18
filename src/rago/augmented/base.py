"""Base classes for the augmented step."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from typeguard import typechecked

from rago.db import DBBase, FaissDB

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from torch import Tensor


@typechecked
class AugmentedBase:
    """Define the base structure for Augmented classes."""

    api_key: str = ''
    model: Optional[Any]
    model_name: str = ''
    db: Any
    top_k: int = 0
    logs: dict[str, Any] = {}  # noqa: RUF012

    # default values to be overwritten by the derived classes
    default_model_name: str = ''
    default_top_k: int = 0

    def __init__(
        self,
        model_name: str = '',
        api_key: str = '',
        db: DBBase = FaissDB(),
        top_k: int = 0,
        logs: dict[str, Any] = {},
    ) -> None:
        """Initialize AugmentedBase."""
        self.db = db
        self.api_key = api_key

        self.top_k = top_k or self.default_top_k
        self.model_name = model_name or self.default_model_name
        self.model = None

        self.logs = logs

        self._validate()
        self._setup()

    def _validate(self) -> None:
        """Raise an error if the initial parameters are not valid."""
        return

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        return

    def get_embedding(
        self, content: list[str]
    ) -> list[Tensor] | npt.NDArray[np.float64] | Tensor:
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
