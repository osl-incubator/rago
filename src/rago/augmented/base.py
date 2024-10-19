""""""

from __future__ import annotations

from abc import abstractclassmethod
from typing import Any, Optional

from rago.db import DBBase


class AugmentedBase:
    model: Optional[Any]
    db: Any
    k: int = -1
    documents: list[str]

    @abstractclassmethod
    def __init__(
        self,
        name: str = 'paraphrase',
        documents: list[str] = [],
        db: FaissDB = FaissDB(),
        k: int = -1,
    ) -> None:
        """Initialize AugmentedBase."""
        ...

    @abstractclassmethod
    def search(self) -> tuple[list[int], list[float]]:
        """
        Search a query into a vector db.

        Return
        ------
        tuple:
            list of indices
            list of distances
        """
        ...

    def run(self, query: str, context: list[str]): ...
