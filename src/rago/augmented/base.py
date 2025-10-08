"""Base classes for the augmented step."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import numpy.typing as npt

from torch import Tensor
from typeguard import typechecked
from typing_extensions import TypeAlias

from rago.augmented.db import DBBase, FaissDB
from rago.base import StepBase
from rago.extensions.cache import Cache
from rago.io import Input, Output

# Define a type alias for embeddings.
EmbeddingType: TypeAlias = Union[
    npt.NDArray[np.float64],
    npt.NDArray[np.float32],
    Tensor,
    list[Tensor],
]

DEFAULT_LOGS: dict[str, Any] = {}


@typechecked
class AugmentedBase(StepBase):
    """Base class for all augmented steps."""

    model: Optional[Any]
    model_name: str = ''
    db: DBBase
    top_k: int = 0

    # Default values to be overwritten by derived classes.
    default_model_name: str = ''
    default_top_k: int = 5

    def __init__(
        self,
        model_name: Optional[str] = None,
        db: DBBase = FaissDB(),
        top_k: Optional[int] = None,
        api_key: str = '',
        api_params: Optional[Dict[str, Any]] = None,
        cache: Optional[Cache] = None,
    ) -> None:
        # super().__init__(api_key=api_key, cache=cache, logs=logs)
        super().__init__()
        self.api_key = api_key
        self.cache = cache
        self.api_params = api_params
        self.db = db
        self.top_k = top_k if top_k is not None else self.default_top_k
        self.model_name = (
            model_name if model_name is not None else self.default_model_name
        )
        self.model = None
        self._validate()
        self._load_optional_modules()
        self._setup()

    def _validate(self) -> None:
        """Override to validate initial parameters."""
        pass

    def _setup(self) -> None:
        """Override to set up the augmented model."""
        pass

    def get_embedding(self, content: list[str]) -> EmbeddingType:
        """Retrieve the embedding for given text."""
        raise Exception('Method not implemented.')

    @abstractmethod
    def search(self, query: str, documents: Any, top_k: int = 0) -> list[str]:
        """Search for the most relevant documents."""
        ...

    def process(self, inp: Input) -> Output:
        """Process the augmented step."""
        query = str(inp.query)
        content = inp.content
        result = self.search(query, content, top_k=self.top_k)
        return Output(data=result)
