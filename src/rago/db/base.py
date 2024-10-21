"""Base classes for database."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any


class DBBase:
    """Base class for vector database."""

    index: Any

    @abstractmethod
    def embed(self, documents: Any) -> None:
        """Embed the documents into the database."""
        ...

    @abstractmethod
    def search(
        self, query_encoded: Any, k: int = 2
    ) -> tuple[list[float], list[int]]:
        """Search a query from documents."""
        ...
