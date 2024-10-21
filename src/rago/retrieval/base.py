"""Base classes for retrieval."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, cast


class RetrievalBase:
    """Base Retrieval class."""

    content: Any

    def __init__(self, sources: Any) -> None:
        self.sources = sources

    @abstractmethod
    def get(self, query: str = '') -> Any:
        """Get the data from the sources."""
        ...


class StringRet(RetrievalBase):
    """String Retrieval class."""

    def get(self, query: str = '') -> list[str]:
        """Get the data from the sources."""
        return cast(List[str], self.sources)
