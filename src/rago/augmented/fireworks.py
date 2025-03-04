"""Classes for augmentation with Fireworks embeddings."""

from __future__ import annotations

from hashlib import sha256
from typing import cast

import numpy as np
import openai  # fireworks client doesnt have query


# embeddings model feature yet
from typeguard import typechecked

from rago.augmented.base import AugmentedBase, EmbeddingType


@typechecked
class FireworksAug(AugmentedBase):
    """Class for augmentation with Fireworks embeddings."""

    default_model_name = 'nomic-ai/nomic-embed-text-v1.5'  # embedding model
    default_top_k = 3

    def _setup(self) -> None:
        """Set up the object with initial parameters."""
        if not self.api_key:
            raise ValueError('API key for Fireworks is required.')
        self.openai_client = openai.OpenAI(
            base_url='https://api.fireworks.ai/inference/v1',
            api_key=self.api_key,
        )

    def get_embedding(self, content: list[str]) -> EmbeddingType:
        """Retrieve the embedding for given texts using the OpenAI client."""
        cache_key = sha256(''.join(content).encode('utf-8')).hexdigest()
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cast(EmbeddingType, cached)

        # Using the OpenAI embeddings API call for fireworks
        response = self.openai_client.embeddings.create(
            model=self.model_name,
            input=content,
        )
        result = np.array(
            [data.embedding for data in response.data], dtype=np.float32
        )
        self._save_cache(cache_key, result)
        return result

    def search(
        self, query: str, documents: list[str], top_k: int = 0
    ) -> list[str]:
        """Search an encoded query into vector database."""
        if not hasattr(self, 'db') or not self.db:
            raise Exception('Vector database (db) is not initialized.')

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
