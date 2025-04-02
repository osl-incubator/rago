"""Base classes for retrieval."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterable, cast

from typeguard import typechecked

from rago.base import RagBase
from rago.retrieval.text_splitter import (
    LangChainTextSplitter,
    TextSplitterBase,
)

DEFAULT_LOGS: dict[str, Any] = {}


@typechecked
class RetrievalBase(RagBase):
    """Base Retrieval class."""

    content: Any
    source: Any
    splitter: TextSplitterBase
    api_params: dict[str, Any] = {}

    def __init__(
        self,
        source: Any,
        splitter: TextSplitterBase = LangChainTextSplitter(
            'RecursiveCharacterTextSplitter'
        ),
        api_key: str = '',
        api_params: dict[str, Any] = {},
    ) -> None:
        super().__init__()
        self.source = source
        self.splitter = splitter
        self.api_key = api_key
        self.api_params = api_params

        self._validate()
        self._setup()

    def _validate(self) -> None:
        """Validate the source. Override if needed."""
        pass

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        pass

    @abstractmethod
    def retrieve(self, query: str = '', source: Any = None) -> Iterable[str]:
        """Get the data from the source."""
        return []
