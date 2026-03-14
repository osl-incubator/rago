"""Base classes for augmentation steps."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any, Optional, Union, cast

import numpy as np
import numpy.typing as npt

from torch import Tensor
from typeguard import typechecked
from typing_extensions import TypeAlias

from rago.augmented.db import DBBase, FaissDB
from rago.base import StepBase, ensure_list
from rago.extensions.cache import Cache
from rago.io import Input, Output

EmbeddingType: TypeAlias = Union[
    npt.NDArray[np.float64],
    npt.NDArray[np.float32],
    Tensor,
    list[Tensor],
]


@typechecked
class AugmentedBase(StepBase):
    """Base class for all augmentation steps."""

    log_name = 'augmented'

    model: Optional[Any]
    model_name: str
    db: DBBase
    top_k: int

    default_model_name: str = ''
    default_top_k: int = 5

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap subclass `search` methods with shared cache handling."""
        super().__init_subclass__(**kwargs)
        method = cls.__dict__.get('search')
        if method is None or getattr(method, '_rago_wrapped', False):
            return

        @wraps(method)
        def wrapped(
            self: AugmentedBase,
            query: str,
            documents: Any,
            top_k: int = 0,
        ) -> list[str]:
            normalized_documents = ensure_list(documents)
            actual_top_k = top_k or self.top_k or self.default_top_k
            cache_key = (
                cls.__name__,
                'search',
                self.model_name,
                query,
                normalized_documents,
                actual_top_k,
            )
            cached = cast(list[str] | None, self._get_cache(cache_key))
            if cached is not None:
                self.logs['cache_hit'] = True
                self.logs['query'] = query
                self.logs['documents'] = normalized_documents
                self.logs['top_k'] = actual_top_k
                self.logs['result'] = cached
                return cached

            typed_method = cast(
                Callable[[AugmentedBase, str, Any, int], list[str]],
                method,
            )
            result = typed_method(
                self, query, normalized_documents, actual_top_k
            )
            self.logs['cache_hit'] = False
            self.logs['query'] = query
            self.logs['documents'] = normalized_documents
            self.logs['top_k'] = actual_top_k
            self.logs['result'] = result
            self._save_cache(cache_key, result)
            return result

        wrapped._rago_wrapped = True  # type: ignore[attr-defined]
        setattr(cls, 'search', wrapped)

    def __init__(
        self,
        model_name: Optional[str] = None,
        db: DBBase | None = None,
        top_k: Optional[int] = None,
        api_key: str = '',
        api_params: dict[str, Any] | None = None,
        cache: Cache | None = None,
        logs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.api_params = api_params or {}
        self.cache = cache
        self.logs = logs or {}
        self.db = db or FaissDB()
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

    def _setup(self) -> None:
        """Override to set up the augmented model."""

    def get_embedding(self, content: list[str]) -> EmbeddingType:
        """Retrieve embeddings for the given texts."""
        raise Exception('Method not implemented.')

    @abstractmethod
    def search(self, query: str, documents: Any, top_k: int = 0) -> list[str]:
        """Search for the most relevant documents."""

    @staticmethod
    def _resolve_retrieved_docs(
        documents: list[str],
        indices: Iterable[str] | Iterable[int],
    ) -> list[str]:
        """Resolve vector DB indices from int or string ids."""
        retrieved_docs: list[str] = []

        for index in indices:
            try:
                resolved_index = int(index)
            except (TypeError, ValueError):
                continue

            if 0 <= resolved_index < len(documents):
                retrieved_docs.append(documents[resolved_index])

        return retrieved_docs

    def process(self, inp: Input) -> Output:
        """Run augmentation against the current pipeline content."""
        query = str(inp.query)
        content = inp.get('content', inp.get('data', inp.get('source')))
        result = self.search(query, ensure_list(content), top_k=self.top_k)
        output = Output.from_input(inp)
        output.content = result
        output.data = result
        return output
