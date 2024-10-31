"""Classes for augmentation with hugging face."""

from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer
from typeguard import typechecked

from rago.augmented.base import AugmentedBase


@typechecked
class HuggingFaceAug(AugmentedBase):
    """Class for augmentation with Hugging Face."""

    default_model_name = 'paraphrase-MiniLM-L6-v2'
    default_k = 2

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        self.model = SentenceTransformer(self.model_name)

    def search(self, query: str, documents: Any, k: int = 0) -> list[str]:
        """Search an encoded query into vector database."""
        if not self.model:
            raise Exception('The model was not created.')

        document_encoded = self.model.encode(documents)
        query_encoded = self.model.encode([query])
        k = k if k > 0 else self.k

        self.db.embed(document_encoded)

        _, indices = self.db.search(query_encoded, k=k)

        retrieved_docs = [documents[i] for i in indices]

        return retrieved_docs
