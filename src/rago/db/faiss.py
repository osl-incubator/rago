"""Module for faiss database."""

from __future__ import annotations

from typing import Any, Iterable

import faiss

from typeguard import typechecked

from rago.db.base import DBBase


@typechecked
class FaissDB(DBBase):
    """Faiss Database."""

    def embed(self, documents: Any) -> None:
        """Embed the documents into the database."""
        self.index = faiss.IndexFlatL2(documents.shape[1])
        self.index.add(documents)

    def search(
        self, query_encoded: Any, top_k: int = 2
    ) -> tuple[Iterable[float], Iterable[int]]:
        """Search an encoded query into vector database."""
        distances, indices = self.index.search(query_encoded, top_k)
        return distances, indices[0]
