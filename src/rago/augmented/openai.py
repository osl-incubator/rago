"""Classes for augmentation with OpenAI embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import openai

from typeguard import typechecked

from rago.augmented.base import AugmentedBase

if TYPE_CHECKING:
    import numpy.typing as npt

    from torch import Tensor


@typechecked
class OpenAIAug(AugmentedBase):
    """Class for augmentation with OpenAI embeddings."""

    default_model_name = 'text-embedding-3-small'
    default_top_k = 2

    def _setup(self) -> None:
        """Set up the object with initial parameters."""
        if not self.api_key:
            raise ValueError('API key for OpenAI is required.')
        openai.api_key = self.api_key
        self.model_name = self.model_name or self.default_model_name
        self.model = openai.OpenAI(api_key=self.api_key)

    def get_embedding(
        self, content: list[str]
    ) -> list[Tensor] | npt.NDArray[np.float64] | Tensor:
        """Retrieve the embedding for a given text using OpenAI API."""
        model = cast(openai.OpenAI, self.model)
        response = model.embeddings.create(
            input=content, model=self.model_name
        )
        result = np.array(response.data[0].embedding)
        return result.reshape(1, result.size)

    def search(
        self, query: str, documents: list[str], top_k: int = 0
    ) -> list[str]:
        """Search an encoded query into vector database."""
        if not hasattr(self, 'db') or not self.db:
            raise Exception('Vector database (db) is not initialized.')

        # Encode the documents and query
        document_encoded = self.get_embedding(documents)
        query_encoded = self.get_embedding([query])
        top_k = top_k or self.top_k or self.default_top_k or 1

        # Embed documents into the vector database and perform the search
        self.db.embed(document_encoded)
        _, indices = self.db.search(query_encoded, top_k=top_k)

        # Retrieve the actual documents based on indices
        retrieved_docs = [documents[i] for i in indices]

        return retrieved_docs
