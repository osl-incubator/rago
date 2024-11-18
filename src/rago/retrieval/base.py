"""Base classes for retrieval."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterable, cast

from typeguard import typechecked

from rago.retrieval.text_splitter import (
    LangChainTextSplitter,
    TextSplitterBase,
)


@typechecked
class RetrievalBase:
    """Base Retrieval class."""

    content: Any
    source: Any
    splitter: TextSplitterBase
    logs: dict[str, Any] = {}  # noqa: RUF012

    def __init__(
        self,
        source: Any,
        splitter: TextSplitterBase = LangChainTextSplitter(
            'RecursiveCharacterTextSplitter'
        ),
        logs: dict[str, Any] = {},
    ) -> None:
        """Initialize the Retrieval class."""
        self.source = source
        self.splitter = splitter

        self.logs = logs

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
