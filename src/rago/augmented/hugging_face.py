"""Classes for augmentation with hugging face."""

from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer

from rago.augmented.base import AugmentedBase
from rago.db import DBBase, FaissDB


class HuggingFaceAug(AugmentedBase):
    """Class for augmentation with Hugging Face."""

    model: Any
    k: int = -1
    db: DBBase

    def __init__(
        self,
        name: str = 'paraphrase',
        db: DBBase = FaissDB(),
        k: int = -1,
    ) -> None:
        """Initialize HuggingFaceAug."""
        if name == 'paraphrase':
            self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        else:
            raise Exception(
                'The Augmented class name {name} is not supported.'
            )

        self.db = db
        self.k = k

    def search(self, query: str, documents: Any, k: int = -1) -> list[str]:
        """Search an encoded query into vector database."""
        document_encoded = self.model.encode(documents)
        query_encoded = self.model.encode([query])
        k = k if k > 0 else self.k

        self.db.embed(document_encoded)

        _, indices = self.db.search(query_encoded, k=k)

        retrieved_docs = [documents[i] for i in indices]

        return retrieved_docs
