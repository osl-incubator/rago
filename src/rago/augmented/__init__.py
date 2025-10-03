"""Rago Augmented package."""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
import numpy.typing as npt

from torch import Tensor
from typeguard import typechecked
from typing_extensions import TypeAlias

from rago.augmented.base import AugmentedBase
from rago.augmented.db import DBBase, FaissDB
from rago.base import ParametersBase, StepBase
from rago.extensions.cache import Cache
from rago.io import Input, Output

# Define a type alias for embeddings.
EmbeddingType: TypeAlias = Union[
    npt.NDArray[np.float64],
    npt.NDArray[np.float32],
    Tensor,
    list[Tensor],
]

DEFAULT_LOGS: dict[str, Any] = {}


@typechecked
@dataclass
class AugmentedParameters(ParametersBase):
    """Parameters for configuring pipeline steps (can be used for augmented and generation)."""

    api_key: str = ''
    model_name: str = ''
    top_k: Optional[int] = 3
    api_params: Optional[Dict[str, Any]] = field(default_factory=dict)


@typechecked
class Augmented(StepBase):
    """
    Public Augmented class for Rago.

    Users instantiate this with parameters (e.g., api_key, model_name, backend,
    top_k, etc.).
    They can combine configuration via the '|' operator or callable syntax.

    When search() is called, the proper specialized augmented instance is
    resolved based on the configuration.
    """

    params: AugmentedParameters

    def __init__(
        self,
        api_key: str = '',
        model_name: str = '',
        backend: str = '',
        engine: str = '',
        top_k: Optional[int] = None,
        api_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.params = AugmentedParameters(
            api_key=api_key,
            model_name=model_name,
            top_k=top_k,
            api_params=api_params,
        )
        self.backend: str = backend.lower() if backend else ''
        self.engine: str = engine.lower() if backend else ''

    def apply(self, parameters: AugmentedParameters) -> None:
        for key, value in parameters.params.items():
            setattr(self, key, value)

    def __add__(self, params: AugmentedParameters) -> Augmented:
        self.apply(params)
        return self

    def __call__(self, **kwargs: Any) -> Augmented:
        params = AugmentedParameters(**kwargs)
        self.apply(params)
        return self

    def _resolve(self) -> AugmentedBase:
        """Resolve and return the specialized augmented instance based on configuration."""
        common_params = self.params.__dict__

        if self.backend == 'cohere':
            from rago.augmented.cohere import CohereAug

            return CohereAug(**common_params)
        elif self.backend == 'fireworks':
            from rago.augmented.fireworks import FireworksAug

            return FireworksAug(**common_params)
        elif self.backend == 'openai':
            from rago.augmented.openai import OpenAIAug

            return OpenAIAug(**common_params)
        elif self.backend == 'sentence_transformers':
            from rago.augmented.sentence_transformer import (
                SentenceTransformerAug,
            )

            return SentenceTransformerAug(**common_params)
        elif self.backend == 'spacy':
            from rago.augmented.spacy import SpaCyAug

            return SpaCyAug(**common_params)
        elif self.backend == 'together':
            from rago.augmented.together import TogetherAug

            return TogetherAug(**common_params)
        else:
            raise Exception(f'Unsupported augmented backend: {self.backend}')

    def search(self, query: str, documents: Any, top_k: int = 0) -> list[str]:
        """Resolve the specialized augmented and delegate the search method."""
        augmented_instance = self._resolve()
        return augmented_instance.search(
            query, documents, top_k=top_k or self.params.top_k
        )

    def process(self, inp: Input) -> Output:
        query = inp.query
        content = inp.content
        top_k = self.params.top_k
        result = self.search(query, content, top_k)
        return Output(content=result)
