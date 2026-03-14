"""Base classes for retrieval steps."""

from __future__ import annotations

from abc import abstractmethod
from functools import wraps
from typing import Any

from typeguard import typechecked

from rago.base import StepBase, ensure_list
from rago.extensions.cache import Cache
from rago.io import Input, Output
from rago.retrieval.text_splitter import (
    LangChainTextSplitter,
    TextSplitterBase,
)


@typechecked
class RetrievalBase(StepBase):
    """Base retrieval class."""

    log_name = 'retrieval'

    content: Any
    source: Any
    splitter: TextSplitterBase
    api_params: dict[str, Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap subclass `retrieve` methods with shared cache handling."""
        super().__init_subclass__(**kwargs)
        method = cls.__dict__.get('retrieve')
        if method is None or getattr(method, '_rago_wrapped', False):
            return

        @wraps(method)
        def wrapped(
            self: RetrievalBase, query: str = '', source: Any = None
        ) -> Any:
            actual_source = self.source if source is None else source
            cache_key = (
                cls.__name__,
                'retrieve',
                query,
                actual_source,
            )
            cached = self._get_cache(cache_key)
            if cached is not None:
                self.logs['cache_hit'] = True
                self.logs['query'] = query
                self.logs['source'] = actual_source
                self.logs['result'] = cached
                return cached

            result = method(self, query, actual_source)
            self.logs['cache_hit'] = False
            self.logs['query'] = query
            self.logs['source'] = actual_source
            self.logs['result'] = result
            self._save_cache(cache_key, result)
            return result

        wrapped._rago_wrapped = True  # type: ignore[attr-defined]
        setattr(cls, 'retrieve', wrapped)

    def __init__(
        self,
        source: Any = None,
        splitter: TextSplitterBase | None = None,
        api_key: str = '',
        api_params: dict[str, Any] | None = None,
        cache: Cache | None = None,
        logs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.source = source
        self.splitter = splitter or LangChainTextSplitter(
            'RecursiveCharacterTextSplitter'
        )
        self.api_key = api_key
        self.api_params = api_params or {}
        self.cache = cache
        self.logs = logs or {}

        self._validate()
        self._setup()

    def _validate(self) -> None:
        """Validate the source. Override if needed."""

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""

    @abstractmethod
    def retrieve(self, query: str = '', source: Any = None) -> list[str]:
        """Get the data from the source."""

    def get(self, query: str = '', source: Any = None) -> list[str]:
        """Backward-compatible alias for `retrieve`."""
        return self.retrieve(query=query, source=source)

    def process(self, inp: Input) -> Output:
        """Resolve the content for downstream steps."""
        source = self.source if self.source is not None else inp.get('source')
        if source is None:
            source = inp.get('content')

        result = ensure_list(self.retrieve(query=inp.query, source=source))
        output = Output.from_input(inp)
        output.content = result
        output.data = result
        return output
