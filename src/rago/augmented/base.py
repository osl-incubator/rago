"""Base classes for the augmented step."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional

from rago.db import DBBase, FaissDB


class AugmentedBase:
    """Define the base structure for Augmented classes."""

    model: Optional[Any]
    db: Any
    k: int = -1
    documents: list[str]

    @abstractmethod
    def __init__(
        self,
        documents: list[str] = [],
        db: DBBase = FaissDB(),
        k: int = -1,
    ) -> None:
        """Initialize AugmentedBase."""
        self.k = k
        self.documents = documents
        self.db = db

    @abstractmethod
    def search(
        self,
        query: str,
        documents: Any,
        k: int = -1,
    ) -> list[str]:
        """Search an encoded query into vector database."""
        ...
