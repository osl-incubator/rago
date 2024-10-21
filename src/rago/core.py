"""Rago is Retrieval Augmented Generation lightweight framework."""

from __future__ import annotations

from rago.augmented.base import AugmentedBase
from rago.generation.base import GenerationBase
from rago.retrieval.base import RetrievalBase


class Rago:
    """RAG class."""

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

    def prompt(self, query: str) -> str:
        """Run the pipeline with for specific prompt."""
        ret_data = self.retrieval.get(query)
        aug_data = self.augmented.search(query, ret_data)
        gen_data = self.generation.generate(query, aug_data)
        return gen_data
