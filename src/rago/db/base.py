"""Base classes for database."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterable

from typeguard import typechecked


class DBBase:
    """Base class for vector database."""

    index: Any

    @abstractmethod
    @typechecked
    def embed(self, documents: Any) -> None:
        """Embed the documents into the database."""
        ...

    @abstractmethod
    @typechecked
    def search(
        self, query_encoded: Any, k: int = 2
    ) -> tuple[Iterable[float], Iterable[int]]:
        """Search a query from documents."""
        ...
