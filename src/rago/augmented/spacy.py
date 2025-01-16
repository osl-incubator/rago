"""Classes for augmentation with SpaCy embeddings."""

from __future__ import annotations

from hashlib import sha256
from typing import List, cast

import numpy as np
import spacy

from typeguard import typechecked

from rago.augmented.base import AugmentedBase, EmbeddingType


@typechecked
class SpaCyAug(AugmentedBase):
    """Class for augmentation with SpaCy embeddings."""

    default_model_name = 'en_core_web_md'
    default_top_k = 3

    def _setup(self) -> None:
        """Set up the object with initial parameters."""
        self.model = spacy.load(self.model_name)

    def get_embedding(self, content: List[str]) -> EmbeddingType:
        """Retrieve the embedding for a given text using SpaCy."""
        cache_key = sha256(''.join(content).encode('utf-8')).hexdigest()
        cached = self._get_cache(cache_key)
        if cached:
            return cast(EmbeddingType, cached)

        model = cast(spacy.language.Language, self.model)
        embeddings = []
        for text in content:
            doc = model(text)
            embeddings.append(doc.vector)
        result = np.array(embeddings)
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
