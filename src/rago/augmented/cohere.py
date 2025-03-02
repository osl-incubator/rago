"""Classes for augmentation with Cohere embeddings."""

from __future__ import annotations

from hashlib import sha256
from typing import cast

import cohere
import numpy as np

from typeguard import typechecked

from rago.augmented.base import AugmentedBase, EmbeddingType


@typechecked
class CohereAug(AugmentedBase):
    """Class for augmentation with Cohere embeddings."""

    default_model_name = 'embed-english-v3.0'  # Cohere's recommended model
    default_top_k = 3

    def _setup(self) -> None:
        """Set up the object with initial parameters."""
        if not self.api_key:
            raise ValueError('API key for Cohere is required.')
        self.model = cohere.Client(self.api_key)

    def get_embedding(self, content: list[str]) -> EmbeddingType:
        """Retrieve the embedding for given texts using Cohere API."""
        cache_key = sha256(''.join(content).encode('utf-8')).hexdigest()
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cast(EmbeddingType, cached)

        model = cast(cohere.Client, self.model)
        response = model.embed(
            texts=content, model=self.model_name, input_type='search_document'
        )

        result = np.array(response.embeddings, dtype=np.float32)

        self._save_cache(cache_key, result)

        return result

    def search(
        self, query: str, documents: list[str], top_k: int = 0
    ) -> list[str]:
        """Search an encoded query into vector database."""
        if not hasattr(self, 'db') or not self.db:
            raise Exception('Vector database (db) is not initialized.')
        document_encoded = self.get_embedding(documents)
        model = cast(cohere.Client, self.model)
        response = model.embed(
            texts=[query], model=self.model_name, input_type='search_query'
        )
        query_encoded = np.array(response.embeddings, dtype=np.float32)

        top_k = top_k or self.top_k or self.default_top_k or 1

        self.db.embed(document_encoded)
        scores, indices = self.db.search(query_encoded, top_k=top_k)

        self.logs['indices'] = indices
        self.logs['scores'] = scores
        self.logs['search_params'] = {
            'query_encoded': query_encoded,
            'top_k': top_k,
        }

        retrieved_docs = [documents[i] for i in indices if i >= 0]

        return retrieved_docs
