"""Base classes for generation steps."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from copy import deepcopy
from functools import wraps
from typing import Any, Optional, Type, cast

import torch

from pydantic import BaseModel
from typeguard import typechecked

from rago.base import StepBase, ensure_list
from rago.extensions.cache import Cache
from rago.io import Input, Output

DEFAULT_API_PARAMS: dict[str, Any] = {}


def _serialize_generation_result(result: str | BaseModel) -> list[str]:
    if isinstance(result, BaseModel):
        return [result.model_dump_json()]
    return [result]


@typechecked
class GenerationBase(StepBase):
    """Generic generation step."""

    log_name = 'generation'

    device_name: str = 'cpu'
    device: torch.device
    model: Any
    model_name: str = ''
    tokenizer: Any
    temperature: float = 0.5
    output_max_length: int = 500
    prompt_template: str = (
        'question: \n```\n{query}\n```\ncontext: ```\n{data}\n```'
    )
    structured_output: Optional[Type[BaseModel]] = None
    api_params: dict[str, Any] = {}
    system_message: str = ''

    default_device_name: str = 'cpu'
    default_model_name: str = ''
    default_temperature: float = 0.0
    default_output_max_length: int = 500
    default_prompt_template: str = (
        'question: \n```\n{query}\n```\ncontext: ```\n{data}\n```'
    )
    default_api_params: dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap subclass `generate` methods with shared cache handling."""
        super().__init_subclass__(**kwargs)
        method = cls.__dict__.get('generate')
        if method is None or getattr(method, '_rago_wrapped', False):
            return

        @wraps(method)
        def wrapped(
            self: GenerationBase,
            query: str,
            data: list[str],
        ) -> str | BaseModel:
            normalized_data = [str(item) for item in ensure_list(data)]
            cache_key = (
                cls.__name__,
                'generate',
                self.model_name,
                query,
                normalized_data,
            )
            cached = cast(str | BaseModel | None, self._get_cache(cache_key))
            if cached is not None:
                self.logs['cache_hit'] = True
                self.logs['query'] = query
                self.logs['data'] = normalized_data
                self.logs['result'] = cached
                return cached

            typed_method = cast(
                Callable[[GenerationBase, str, list[str]], str | BaseModel],
                method,
            )
            result = typed_method(self, query, normalized_data)
            self.logs['cache_hit'] = False
            self.logs['query'] = query
            self.logs['data'] = normalized_data
            self.logs['result'] = result
            self._save_cache(cache_key, result)
            return result

        wrapped._rago_wrapped = True  # type: ignore[attr-defined]
        setattr(cls, 'generate', wrapped)

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        prompt_template: str = '',
        output_max_length: int = 500,
        device: str = 'auto',
        structured_output: Optional[Type[BaseModel]] = None,
        system_message: str = '',
        api_params: dict[str, Any] = DEFAULT_API_PARAMS,
        api_key: str = '',
        cache: Cache | None = None,
        logs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.cache = cache
        self.logs = logs or {}

        self.model_name = (
            model_name if model_name is not None else self.default_model_name
        )
        self.output_max_length = (
            output_max_length or self.default_output_max_length
        )
        self.temperature = (
            temperature
            if temperature is not None
            else self.default_temperature
        )
        self.prompt_template = prompt_template or self.default_prompt_template
        self.structured_output = structured_output
        if api_params is DEFAULT_API_PARAMS:
            api_params = deepcopy(self.default_api_params or {})

        self.system_message = system_message
        self.api_params = api_params

        if device not in ['cpu', 'cuda', 'auto']:
            raise Exception(
                f'Device {device} not supported. Options: cpu, cuda, auto.'
            )

        cuda_available = torch.cuda.is_available()
        self.device_name = (
            'cpu' if device == 'cpu' or not cuda_available else 'cuda'
        )
        self.device = torch.device(self.device_name)

        self._validate()
        self._load_optional_modules()
        self._setup()

    def _validate(self) -> None:
        """Raise an error if the initial parameters are not valid."""

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""

    def _format_prompt(self, query: str, data: list[str]) -> str:
        joined = ' '.join(data)
        return self.prompt_template.format(
            query=query,
            data=joined,
            context=joined,
        )

    @abstractmethod
    def generate(
        self,
        query: str,
        data: list[str],
    ) -> str | BaseModel:
        """Generate text with optional language parameter."""

    def process(self, inp: Input) -> Output:
        """Generate a result from the current pipeline content."""
        query = str(inp.query)
        data = [
            str(item)
            for item in ensure_list(
                inp.get('content', inp.get('data', inp.get('source')))
            )
        ]
        result = self.generate(query, data)
        output = Output.from_input(inp)
        output.result = result
        output.content = _serialize_generation_result(result)
        output.data = output.content
        return output
