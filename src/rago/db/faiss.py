""""""

from __future__ import annotations

from rago.db.base import DBBase


class FaissDB(DBBase):
    """"""

    def embed(self, documents: Any) -> None:
        """
        Parameters
        ----------
        documents: encoded documents
        """
        self.index = faiss.IndexFlatL2(documents.shape[1])
        self.index.add(documents)

    def search(
        self, query: str, model: Any, k: int = 2
    ) -> tuple[list[int], list[float]]:
        query_embedding = model.encode([query])

        # Step 4: Perform the search
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices
