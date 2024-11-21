"""Rago is Retrieval Augmented Generation lightweight framework."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from typeguard import typechecked

from rago.augmented.base import AugmentedBase
from rago.generation.base import GenerationBase
from rago.retrieval.base import RetrievalBase


@typechecked
class Rago:
    """RAG class."""

    retrieval: RetrievalBase
    augmented: AugmentedBase
    generation: GenerationBase

    logs: dict[str, dict[str, Any]]

    def __init__(
        self,
        retrieval: RetrievalBase,
        augmented: AugmentedBase,
        generation: GenerationBase,
    ) -> None:
        """Initialize the RAG structure.

        Parameters
        ----------
        retrieval : RetrievalBase
            The retrieval component used to fetch relevant data based
            on the query.
        augmented : AugmentedBase
            The augmentation module responsible for enriching the
            retrieved data.
        generation : GenerationBase
            The text generation model used to generate a response based
            on the query and augmented data.
        """
        self.retrieval = retrieval
        self.augmented = augmented
        self.generation = generation
        self.logs: dict[str, dict[str, Any]] = {
            'retrieval': retrieval.logs,
            'augmented': augmented.logs,
            'generation': generation.logs,
        }

    def prompt(self, query: str, device: str = 'auto') -> str | BaseModel:
        """Run the pipeline for a specific prompt.

        Parameters
        ----------
        query : str
            The query or prompt from the user.
        device : str (default 'auto')
            Device for generation (e.g., 'auto', 'cpu', 'cuda'), by
            default 'auto'.

        Returns
        -------
        str
            Generated text based on the query and augmented data.
        """
        ret_data = self.retrieval.get(query)
        self.logs['retrieval']['result'] = ret_data

        aug_data = self.augmented.search(query, ret_data)
        self.logs['augmented']['result'] = aug_data

        gen_data = self.generation.generate(query, context=aug_data)
        self.logs['generation']['result'] = gen_data

        return gen_data
