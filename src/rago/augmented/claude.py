"""ClaudeAug: Augmentation with Anthropic Claude embeddings via Amazon Bedrock."""

from __future__ import annotations

from hashlib import sha256
from typing import cast, List, Dict, Any

import boto3
import json
import numpy as np

from typeguard import typechecked

from rago.augmented.base import AugmentedBase, EmbeddingType

@typechecked
class ClaudeAug(AugmentedBase):
    """Class for augmentation with Anthropic Claude embeddings via Bedrock."""

    default_model_name = 'anthropic.claude-v2'
    default_top_k = 3

    def _setup(self) -> None:
        """Set up the object with initial parameters."""
        if not self.api_key:
            raise ValueError("Claude API key is required.")
        self.client = boto3.client("bedrock-runtime")

    def get_embedding(self, content: List[str]) -> EmbeddingType:
        """Retrieve embeddings from Claude via Amazon Bedrock."""
        cache_key = sha256(''.join(content).encode('utf-8')).hexdigest()
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cast(EmbeddingType, cached)

        model_params: Dict[str, Any] = {
            "modelId": self.model_name,
            "prompt": "Generate a numerical vector representation for semantic similarity: " + ' '.join(content),
            "temperature": 0.0,
        }

        response = self.client.invoke_model(
            body=json.dumps(model_params),
            modelId=self.model_name
        )
        result = json.loads(response['body'].read().decode('utf-8'))
        embedding = np.array(result.get("embeddings", []), dtype=np.float32)

        if embedding.size == 0:
            raise ValueError("Failed to generate embeddings from Claude API.")

        self._save_cache(cache_key, embedding)
        return embedding

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
