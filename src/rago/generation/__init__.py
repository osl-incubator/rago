"""RAG Generation package."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, Union

import torch

from pydantic import BaseModel
from typeguard import typechecked

from rago.base import ParametersBase, StepBase
from rago.generation.base import GenerationBase
from rago.generation.cohere import CohereGen
from rago.generation.deepseek import DeepSeekGen
from rago.generation.fireworks import FireworksGen
from rago.generation.gemini import GeminiGen
from rago.generation.groq import GroqGen
from rago.generation.hugging_face import HuggingFaceGen
from rago.generation.hugging_face_inf import HuggingFaceInfGen
from rago.generation.llama import LlamaGen, OllamaGen, OllamaOpenAIGen
from rago.generation.openai import OpenAIGen
from rago.generation.phi import PhiGen
from rago.generation.together import TogetherGen
from rago.io import Input, Output

__all__ = [
    'CohereGen',
    'DeepSeekGen',
    'FireworksGen',
    'GeminiGen',
    'GenerationBase',
    'GroqGen',
    'HuggingFaceGen',
    'HuggingFaceInfGen',
    'LlamaGen',
    'OllamaGen',
    'OllamaOpenAIGen',
    'OpenAIGen',
    'PhiGen',
    'TogetherGen',
]

# ---------------------------------------
# GenerationParameters (inherits from StepBase)
# ---------------------------------------
@typechecked
class GenerationParameters(ParametersBase):
    """Parameters for configuring generation steps."""

    api_key: str = ''
    model_name: str = ''
    backend: str = ''
    engine: str = ''
    temperature: float = 0.0
    prompt_template: str = ''
    output_max_length: int = 500
    structured_output: Optional[Type[BaseModel]] = None
    api_params: Optional[Dict[str, Any]] = {}


@typechecked
class Generation(StepBase):
    """
    Public Generation class for Rago.

    Users instantiate Generation with desired configuration parameters such as:

        api_key, model_name, backend, engine, temperature, etc.

    They can combine these with a Parameters object via '+' or callable syntax.

    When generate() is called, the proper specialized generation instance is resolved
    based on the configuration.
    """

    params: GenerationParameters

    def __init__(
        self,
        api_key: str = '',
        model_name: str = '',
        backend: str = '',
        engine: str = '',
        temperature: float = 0.0,
        prompt_template: str = '',
        output_max_length: int = 500,
        structured_output: Optional[Type[BaseModel]] = None,
        api_params: Optional[Dict[str, Any]] = {},
    ) -> None:
        self.params = GenerationParameters(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            prompt_template=prompt_template,
            output_max_length=output_max_length,
            structured_output=structured_output,
            api_params=api_params,
        )
        # Additional parameters to select the proper backend/engine.
        self.backend: str = backend.lower() if backend else ''
        self.engine: Optional[str] = engine.lower() if engine else None

    def apply(self, parameters: GenerationParameters) -> None:
        for key, value in parameters.params.items():
            setattr(self, key, value)

    def __add__(self, params: GenerationParameters) -> Generation:
        self.params = params
        return self

    def __call__(self, **kwargs: Any) -> Generation:
        self.params = GenerationParameters(**kwargs)
        return self

    def _resolve(self) -> GenerationBase:
        """Resolve and return the specialized generation instance based on configuration."""
        # Prepare common parameters to pass to the specialized generator.
        common_params = self.params

        if self.backend == 'openai':
            return OpenAIGen(**common_params)
        elif self.backend == 'llama':
            if self.engine is None or self.engine == 'huggingface':
                return LlamaGen(**common_params)
            elif self.engine == 'ollama':
                raise NotImplementedError('Ollama engine not implemented.')
            else:
                raise Exception(f'Unsupported engine for llama: {self.engine}')
        elif self.backend == 'cohere':
            return CohereGen(**common_params)
        elif self.backend == 'deepseek':
            return DeepSeekGen(**common_params)
        elif self.backend == 'fireworks':
            return FireworksGen(**common_params)
        elif self.backend == 'gemini':
            return GeminiGen(**common_params)
        elif self.backend == 'groq':
            return GroqGen(**common_params)
        elif self.backend == 'huggingface':
            return HuggingFaceGen(**common_params)
        elif self.backend == 'phi':
            return PhiGen(**common_params)
        elif self.backend == 'together':
            return TogetherGen(**common_params)
        else:
            raise Exception(f'Unsupported backend: {self.backend}')

    def process(self, inp: Input) -> Output:
        """Resolve the specialized generator and delegate the generate() call."""
        query = inp.query
        data = inp.content
        generator_instance = self._resolve()
        result = generator_instance.generate(query, data)
        return Output(result=result)
