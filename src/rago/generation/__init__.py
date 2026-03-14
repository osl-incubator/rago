"""Composable generation APIs for Rago."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Type

from pydantic import BaseModel
from typeguard import typechecked

from rago.base import ParametersBase, StepBase, config_to_dict, ensure_list
from rago.generation.base import GenerationBase
from rago.io import Input, Output

__all__ = [
    'CohereGen',
    'DeepSeekGen',
    'FireworksGen',
    'GeminiGen',
    'Generation',
    'GenerationBase',
    'GenerationParameters',
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


@typechecked
class GenerationParameters(ParametersBase):
    """Parameters for configuring generation steps."""


@typechecked
class Generation(StepBase):
    """Public generation wrapper that resolves a concrete backend lazily."""

    log_name = 'generation'

    def __init__(
        self,
        api_key: str = '',
        model_name: str = '',
        backend: str = '',
        engine: str = '',
        temperature: float = 0.0,
        prompt_template: str = '',
        output_max_length: int = 500,
        structured_output: Type[BaseModel] | None = None,
        api_params: dict[str, Any] | None = None,
        device: str = 'auto',
        system_message: str = '',
        cache: Any = None,
        logs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend.lower() if backend else ''
        self.engine = engine.lower() if engine else ''
        self.params = GenerationParameters(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            prompt_template=prompt_template,
            output_max_length=output_max_length,
            structured_output=structured_output,
            api_params=api_params or {},
            device=device,
            system_message=system_message,
        )
        self.cache = cache
        self.logs = logs if logs is not None else {}

    def __call__(self, **kwargs: Any) -> Generation:
        """Update this wrapper with additional generation parameters."""
        self.apply(GenerationParameters(**kwargs))
        return self

    def apply(self, parameters: Any) -> None:
        """Apply declarative configuration to the generation wrapper."""
        super().apply(parameters)
        for key, value in config_to_dict(parameters).items():
            if key == 'backend' and isinstance(value, str):
                self.backend = value.lower()
            elif key == 'engine' and isinstance(value, str):
                self.engine = value.lower()
            else:
                self.params.params[key] = value

    def _resolve(self) -> GenerationBase:
        config = deepcopy(self.params.params)
        if self.cache is not None:
            config['cache'] = self.cache
        config['logs'] = self.logs

        if self.backend == 'openai':
            from rago.generation.openai import OpenAIGen

            return OpenAIGen(**config)
        if self.backend == 'llama':
            if not self.engine or self.engine == 'huggingface':
                from rago.generation.llama import LlamaGen

                return LlamaGen(**config)
            if self.engine == 'ollama':
                from rago.generation.llama import OllamaGen

                return OllamaGen(**config)
            if self.engine == 'openai':
                from rago.generation.llama import OllamaOpenAIGen

                return OllamaOpenAIGen(**config)
            raise Exception(f'Unsupported engine for llama: {self.engine}')
        if self.backend == 'ollama':
            from rago.generation.llama import OllamaGen

            return OllamaGen(**config)
        if self.backend == 'ollama-openai':
            from rago.generation.llama import OllamaOpenAIGen

            return OllamaOpenAIGen(**config)
        if self.backend == 'cohere':
            from rago.generation.cohere import CohereGen

            return CohereGen(**config)
        if self.backend == 'deepseek':
            from rago.generation.deepseek import DeepSeekGen

            return DeepSeekGen(**config)
        if self.backend == 'fireworks':
            from rago.generation.fireworks import FireworksGen

            return FireworksGen(**config)
        if self.backend == 'gemini':
            from rago.generation.gemini import GeminiGen

            return GeminiGen(**config)
        if self.backend == 'groq':
            from rago.generation.groq import GroqGen

            return GroqGen(**config)
        if self.backend == 'huggingface':
            from rago.generation.hugging_face import HuggingFaceGen

            return HuggingFaceGen(**config)
        if self.backend == 'huggingface-inference':
            from rago.generation.hugging_face_inf import HuggingFaceInfGen

            return HuggingFaceInfGen(**config)
        if self.backend == 'phi':
            from rago.generation.phi import PhiGen

            return PhiGen(**config)
        if self.backend == 'together':
            from rago.generation.together import TogetherGen

            return TogetherGen(**config)
        raise Exception(f'Unsupported backend: {self.backend}')

    def generate(self, query: str, data: list[str]) -> str | BaseModel:
        """Resolve the concrete generator and run generation."""
        generator_instance = self._resolve()
        normalized_data = [str(item) for item in ensure_list(data)]
        return generator_instance.generate(query, normalized_data)

    def process(self, inp: Input) -> Output:
        """Process the current pipeline content with generation."""
        generator_instance = self._resolve()
        return generator_instance.process(inp)


def __getattr__(name: str) -> Any:
    if name == 'OpenAIGen':
        from rago.generation.openai import OpenAIGen

        return OpenAIGen
    if name == 'GeminiGen':
        from rago.generation.gemini import GeminiGen

        return GeminiGen
    if name == 'HuggingFaceGen':
        from rago.generation.hugging_face import HuggingFaceGen

        return HuggingFaceGen
    if name == 'HuggingFaceInfGen':
        from rago.generation.hugging_face_inf import HuggingFaceInfGen

        return HuggingFaceInfGen
    if name == 'LlamaGen':
        from rago.generation.llama import LlamaGen

        return LlamaGen
    if name == 'OllamaGen':
        from rago.generation.llama import OllamaGen

        return OllamaGen
    if name == 'OllamaOpenAIGen':
        from rago.generation.llama import OllamaOpenAIGen

        return OllamaOpenAIGen
    if name == 'CohereGen':
        from rago.generation.cohere import CohereGen

        return CohereGen
    if name == 'DeepSeekGen':
        from rago.generation.deepseek import DeepSeekGen

        return DeepSeekGen
    if name == 'FireworksGen':
        from rago.generation.fireworks import FireworksGen

        return FireworksGen
    if name == 'TogetherGen':
        from rago.generation.together import TogetherGen

        return TogetherGen
    if name == 'GroqGen':
        from rago.generation.groq import GroqGen

        return GroqGen
    if name == 'PhiGen':
        from rago.generation.phi import PhiGen

        return PhiGen
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
