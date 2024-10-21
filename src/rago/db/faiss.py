"""Module for faiss database."""

from __future__ import annotations

from typing import Any

import faiss

from rago.db.base import DBBase


class FaissDB(DBBase):
    """Faiss Database."""

    def embed(self, documents: Any) -> None:
        """Embed the documents into the database."""
        self.index = faiss.IndexFlatL2(documents.shape[1])
        self.index.add(documents)

    def search(
        self, query_encoded: Any, k: int = 2
    ) -> tuple[list[float], list[int]]:
        """Search an encoded query into vector database."""
        distances, indices = self.index.search(query_encoded, k)
        return distances, indices[0]
