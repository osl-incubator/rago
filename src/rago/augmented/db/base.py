"""Base classes for database."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterable

from typeguard import typechecked


@typechecked
class DBBase:
    """Base class for vector database."""

    index: Any

    @abstractmethod
    def embed(self, documents: Any) -> None:
        """Embed the documents into the database."""
        ...

    @abstractmethod
    def search(
        self, query_encoded: Any, top_k: int = 2
    ) -> tuple[Iterable[float], Iterable[int]]:
        """Search a query from documents."""
        ...
