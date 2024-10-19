"""Rago is Retrieval Augmented Generation lightweight framework."""

from __future__ import annotations

import faiss

from rago.augmented.base import AugmentedBase
from rago.generation.base import GenerationBase
from rago.retrieval.base import RetrievalBase


class Rago:
    retrieval: RetrievalBase
    augmented: AugmentedBase
    generation: GenerationBase

    def __init__(
        self,
        retrieval: RetrievalBase,
        augmented: AugmentedBase,
        generation: GenerationBase,
    ) -> None:
        """Initialize the RAG structure."""
        self.retrieval = retrieval
        self.augmented = augmented
        self.generation = generation

    def prompt(self, query: str) -> str: ...
