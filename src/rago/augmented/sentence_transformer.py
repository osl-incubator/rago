"""Classes for augmentation with hugging face."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from typeguard import typechecked

from rago._optional import require_dependency
from rago.augmented.base import AugmentedBase, EmbeddingType

if TYPE_CHECKING:
    from sentence_transformer import SentenceTransformer


@typechecked
class SentenceTransformerAug(AugmentedBase):
    """Class for augmentation with Hugging Face."""

    default_model_name = 'paraphrase-MiniLM-L12-v2'
    default_top_k = 3

    def _load_optional_modules(self) -> None:
        self._sentence_transformers = require_dependency(
            'sentence_transformers',
            extra='sentence_transformers',
            context='HF Transformer',
        )

        self._SentenceTransformer = (
            self._sentence_transformers.SentenceTransformer
        )

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        self.model = self._SentenceTransformer(self.model_name)

    def get_embedding(self, content: list[str]) -> EmbeddingType:
        """Retrieve the embedding for a given text using OpenAI API."""
        model = cast('SentenceTransformer', self.model)
        return cast(EmbeddingType, model.encode(content))

    def search(self, query: str, documents: Any, top_k: int = 0) -> list[str]:
        """Search an encoded query into vector database."""
        if not self.model:
            raise Exception('The model was not created.')

        document_encoded = self.get_embedding(documents)
        query_encoded = self.get_embedding([query])
        top_k = top_k or self.top_k or self.default_top_k or 1

        self.db.embed(document_encoded)

        scores, indices = self.db.search(query_encoded, top_k=top_k)

        retrieved_docs = [documents[i] for i in indices]

        self.logs['indices'] = indices
        self.logs['scores'] = scores
        self.logs['search_params'] = {
            'query_encoded': query_encoded,
            'top_k': top_k,
        }

        return retrieved_docs
