"""Composable augmentation APIs for Rago."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from typeguard import typechecked

from rago.augmented.base import AugmentedBase
from rago.base import ParametersBase, StepBase, config_to_dict
from rago.io import Input, Output

__all__ = [
    'Augmented',
    'AugmentedBase',
    'AugmentedParameters',
    'CohereAug',
    'FireworksAug',
    'OpenAIAug',
    'SentenceTransformerAug',
    'SpaCyAug',
    'TogetherAug',
]


@typechecked
class AugmentedParameters(ParametersBase):
    """Parameters for configuring augmentation steps."""


@typechecked
class Augmented(StepBase):
    """Public augmentation wrapper that resolves a concrete backend lazily."""

    log_name = 'augmented'

    def __init__(
        self,
        api_key: str = '',
        model_name: str = '',
        backend: str = '',
        engine: str = '',
        top_k: int | None = None,
        api_params: dict[str, Any] | None = None,
        db: Any = None,
        cache: Any = None,
        logs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend.lower() if backend else ''
        self.engine = engine.lower() if engine else ''
        self.params = AugmentedParameters(
            api_key=api_key,
            model_name=model_name,
            top_k=top_k,
            api_params=api_params or {},
        )
        self.db = db
        self.cache = cache
        self.logs = logs if logs is not None else {}

    def __call__(self, **kwargs: Any) -> Augmented:
        """Update this wrapper with additional augmentation parameters."""
        self.apply(AugmentedParameters(**kwargs))
        return self

    def apply(self, parameters: Any) -> None:
        """Apply declarative configuration to the augmentation wrapper."""
        super().apply(parameters)
        for key, value in config_to_dict(parameters).items():
            if key == 'backend' and isinstance(value, str):
                self.backend = value.lower()
            elif key == 'engine' and isinstance(value, str):
                self.engine = value.lower()
            elif key == 'db':
                self.db = value
            else:
                self.params.params[key] = value

    def _resolve(self) -> AugmentedBase:
        config = deepcopy(self.params.params)
        if self.db is not None:
            config['db'] = self.db
        if self.cache is not None:
            config['cache'] = self.cache
        config['logs'] = self.logs

        if self.backend == 'cohere':
            from rago.augmented.cohere import CohereAug

            return CohereAug(**config)
        if self.backend == 'fireworks':
            from rago.augmented.fireworks import FireworksAug

            return FireworksAug(**config)
        if self.backend == 'openai':
            from rago.augmented.openai import OpenAIAug

            return OpenAIAug(**config)
        if self.backend == 'sentence_transformers':
            from rago.augmented.sentence_transformer import (
                SentenceTransformerAug,
            )

            return SentenceTransformerAug(**config)
        if self.backend == 'spacy':
            from rago.augmented.spacy import SpaCyAug

            return SpaCyAug(**config)
        if self.backend == 'together':
            from rago.augmented.together import TogetherAug

            return TogetherAug(**config)
        raise Exception(f'Unsupported augmented backend: {self.backend}')

    def search(self, query: str, documents: Any, top_k: int = 0) -> list[str]:
        """Resolve the concrete augmenter and run search."""
        augmented_instance = self._resolve()
        return augmented_instance.search(query, documents, top_k=top_k)

    def process(self, inp: Input) -> Output:
        """Process the current pipeline content with augmentation."""
        content = inp.get('content', inp.get('data', inp.get('source')))
        result = self.search(
            inp.query,
            content,
            top_k=self.params.params.get('top_k', 0) or 0,
        )
        output = Output.from_input(inp)
        output.content = result
        output.data = result
        return output


def __getattr__(name: str) -> Any:
    if name == 'CohereAug':
        from rago.augmented.cohere import CohereAug

        return CohereAug
    if name == 'FireworksAug':
        from rago.augmented.fireworks import FireworksAug

        return FireworksAug
    if name == 'OpenAIAug':
        from rago.augmented.openai import OpenAIAug

        return OpenAIAug
    if name == 'SentenceTransformerAug':
        from rago.augmented.sentence_transformer import SentenceTransformerAug

        return SentenceTransformerAug
    if name == 'SpaCyAug':
        from rago.augmented.spacy import SpaCyAug

        return SpaCyAug
    if name == 'TogetherAug':
        from rago.augmented.together import TogetherAug

        return TogetherAug
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
