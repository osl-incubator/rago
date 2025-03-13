"""ChromaDB implementation for vector database."""

from typing import Any, List, Tuple

import numpy as np

from chromadb.api import ClientAPI

from rago.augmented.db.base import DBBase


class ChromaDB(DBBase):
    """ChromaDB implementation for vector database."""

    def __init__(
        self,
        client: ClientAPI,
        collection_name: str = 'rago',
    ) -> None:
        """Initialize ChromaDB."""
        self.client = client
        self.collection_name = collection_name
        self._setup()

    def _setup(self) -> None:
        """Set up ChromaDB client and collection."""
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        self.index = self.collection

    def embed(self, documents: Any) -> None:
        """Embed the documents into the database."""
        if not isinstance(documents, tuple) or len(documents) != 2:
            raise ValueError(
                'documents format must be: (List[str], List[List[float]])'
            )

        documents_list: List[str] = documents[0]
        embeddings_list: List[List[float]] = documents[1]

        # Convert embeddings to numpy array
        embeddings = np.array(embeddings_list, dtype=np.float32)

        self.collection.add(
            documents=documents_list,
            embeddings=embeddings,
            ids=[str(i) for i in range(len(documents_list))],
        )

    def search(
        self, query_encoded: Any, top_k: int = 2
    ) -> Tuple[List[float], List[str]]:
        """Search a query from documents."""
        # Convert query_encoded to numpy array
        query_encoded_np = np.array([query_encoded], dtype=np.float32)

        results = self.collection.query(
            query_embeddings=query_encoded_np.tolist(),
            n_results=top_k,
        )

        # Check if keys exist before accessing them
        distances = results.get('distances', [[]])
        ids = results.get('ids', [[]])

        # Ensure distances and ids are not None before indexing
        distances_list: List[float] = distances[0] if distances else []
        ids_list: List[str] = ids[0] if ids else []

        return distances_list, ids_list
