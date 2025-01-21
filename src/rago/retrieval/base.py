"""Base classes for retrieval."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterable, Optional, cast

from typeguard import typechecked

from rago.base import RagoBase
from rago.extensions.cache import Cache
from rago.retrieval.text_splitter import (
    LangChainTextSplitter,
    TextSplitterBase,
)

DEFAULT_LOGS: dict[str, Any] = {}


@typechecked
class RetrievalBase(RagoBase):
    """Base Retrieval class."""

    content: Any
    source: Any
    splitter: TextSplitterBase

    def __init__(
        self,
        source: Any,
        splitter: TextSplitterBase = LangChainTextSplitter(
            'RecursiveCharacterTextSplitter'
        ),
        api_key: str = '',
        cache: Optional[Cache] = None,
        logs: dict[str, Any] = DEFAULT_LOGS,
    ) -> None:
        """Initialize the Retrieval class."""
        if logs is DEFAULT_LOGS:
            logs = {}
        super().__init__(api_key=api_key, cache=cache, logs=logs)
        self.source = source
        self.splitter = splitter

        self._validate()
        self._setup()

    def _validate(self) -> None:
        """Validate if the source is valid, otherwise raises an exception."""
        return None

    def _setup(self) -> None:
        """Set up the object with the giving initial parameters."""
        return None

    @abstractmethod
    def get(self, query: str = '') -> Iterable[str]:
        """Get the data from the source."""
        return []


@typechecked
class StringRet(RetrievalBase):
    """
    String Retrieval class.

    This is a very generic class that assumes that the input (source) is
    already a list of strings.
    """

    def get(self, query: str = '') -> Iterable[str]:
        """Get the data from the sources."""
        return cast(list[str], self.source)
