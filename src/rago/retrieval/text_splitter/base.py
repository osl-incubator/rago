"""The base classes for text splitter."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterable


class TextSplitterBase:
    """The base text splitter class."""

    chunk_size: int = 500
    chunk_overlap: int = 100
    splitter_name: str = ''
    splitter: Any = None

    # defaults

    default_chunk_size: int = 500
    default_chunk_overlap: int = 100
    default_splitter_name: str = ''
    default_splitter: Any = None

    def __init__(
        self,
        splitter_name: str = '',
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:
        """Initialize the text splitter class."""
        self.chunk_size = chunk_size or self.default_chunk_size
        self.chunk_overlap = chunk_overlap or self.default_chunk_overlap
        self.splitter_name = splitter_name or self.default_splitter_name

        self._validate()
        self._setup()

    def _validate(self) -> None:
        """Validate if the initial parameters are valid."""
        return

    def _setup(self) -> None:
        """Set up the object according to the given parameters."""
        return

    @abstractmethod
    def split(self, text: str) -> Iterable[str]:
        """Split a text into chunks."""
        return []
