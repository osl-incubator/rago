"""Classes for augmentation with hugging face."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from sentence_transformers import SentenceTransformer
from typeguard import typechecked

from rago.augmented.base import AugmentedBase

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from torch import Tensor


@typechecked
class SentenceTransformerAug(AugmentedBase):
    """Class for augmentation with Hugging Face."""

    default_model_name = 'paraphrase-MiniLM-L12-v2'
    default_top_k = 3

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        self.model = SentenceTransformer(self.model_name)

    def get_embedding(
        self, content: list[str]
    ) -> list[Tensor] | npt.NDArray[np.float64] | Tensor:
        """Retrieve the embedding for a given text using OpenAI API."""
        model = cast(SentenceTransformer, self.model)
        return model.encode(content)

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
