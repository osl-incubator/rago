"""Classes for augmentation with OpenAI embeddings."""

from __future__ import annotations

from hashlib import sha256
from typing import cast

import numpy as np
import openai

from typeguard import typechecked

from rago.augmented.base import AugmentedBase, EmbeddingType


@typechecked
class OpenAIAug(AugmentedBase):
    """Class for augmentation with OpenAI embeddings."""

    default_model_name = 'text-embedding-3-small'
    default_top_k = 3

    def _setup(self) -> None:
        """Set up the object with initial parameters."""
        if not self.api_key:
            raise ValueError('API key for OpenAI is required.')
        openai.api_key = self.api_key
        self.model = openai.OpenAI(api_key=self.api_key)

    def get_embedding(self, content: list[str]) -> EmbeddingType:
        """Retrieve the embedding for a given text using OpenAI API."""
        cache_key = sha256(''.join(content).encode('utf-8')).hexdigest()
        cached = self._get_cache(cache_key)
        if cached:
            return cast(EmbeddingType, cached)

        model = cast(openai.OpenAI, self.model)
        response = model.embeddings.create(
            input=content, model=self.model_name
        )
        result = np.array(response.data[0].embedding)
        result = result.reshape(1, result.size)

        self._save_cache(cache_key, result)

        return result

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
