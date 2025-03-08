"""MistralAug: Augmentation with Mistral AI embeddings."""

from __future__ import annotations

from hashlib import sha256
from typing import cast, List, Dict, Any

import os
import numpy as np
from mistral.client import Mistral
from mistral.models.embeddings import EmbeddingRequest

from typeguard import typechecked

from rago.augmented.base import AugmentedBase, EmbeddingType

@typechecked
class MistralAug(AugmentedBase):
    """Class for augmentation with Mistral AI embeddings."""

    default_model_name = 'mistral-embed'
    default_top_k = 3

    def _setup(self) -> None:
        """Set up the object with initial parameters."""
        if not self.api_key:
            raise ValueError("Mistral API key is required.")
        self.client = Mistral(api_key=self.api_key)

    def get_embedding(self, content: List[str]) -> EmbeddingType:
        """Retrieve embeddings from Mistral AI."""
        cache_key = sha256(''.join(content).encode('utf-8')).hexdigest()
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cast(EmbeddingType, cached)

        request = EmbeddingRequest(model=self.model_name, input=content)
        response = self.client.embeddings(request)
        embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]

        if not embeddings:
            raise ValueError("No embeddings returned from Mistral API.")

        embeddings_array = np.vstack(embeddings)
        self._save_cache(cache_key, embeddings_array)
        return embeddings_array

    def search(self, query: str, documents: List[str], top_k: int = 0) -> List[str]:
        """Search an encoded query into a vector database."""
        if not hasattr(self, 'db') or not self.db:
            raise RuntimeError('Vector database (db) is not initialized.')

        document_encoded = self.get_embedding(documents)
        query_encoded = self.get_embedding([query])

        top_k = top_k or self.top_k or self.default_top_k or 1

        self.db.embed(document_encoded)
        scores, indices = self.db.search(query_encoded, top_k=top_k)

        self.logs['indices'] = indices
        self.logs['scores'] = scores
        self.logs['search_params'] = {"query_encoded": query_encoded, "top_k": top_k}

        return [documents[i] for i in indices if i >= 0]
