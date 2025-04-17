"""Base classes for generation."""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import Any, Optional, Type

import torch

from pydantic import BaseModel
from typeguard import typechecked

from rago.base import RagoBase
from rago.extensions.cache import Cache

DEFAULT_LOGS: dict[str, Any] = {}
DEFAULT_API_PARAMS: dict[str, Any] = {}


@typechecked
class GenerationBase(RagoBase):
    """Generic Generation class."""

    device_name: str = 'cpu'
    device: torch.device
    model: Any
    model_name: str = ''
    tokenizer: Any
    temperature: float = 0.5
    output_max_length: int = 500
    prompt_template: str = (
        'question: \n```\n{query}\n```\ncontext: ```\n{context}\n```'
    )
    structured_output: Optional[Type[BaseModel]] = None
    api_params: dict[str, Any] = {}
    system_message: str = ''

    # default parameters that can be overwritten by the derived class
    default_device_name: str = 'cpu'
    default_model_name: str = ''
    default_temperature: float = 0.5
    default_output_max_length: int = 500
    default_prompt_template: str = (
        'question: \n```\n{query}\n```\ncontext: ```\n{context}\n```'
    )
    default_api_params: dict[str, Any] = {}

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
        cache: Optional[Cache] = None,
        logs: dict[str, Any] = DEFAULT_LOGS,
    ) -> None:
        """Initialize Generation class."""
        if logs is DEFAULT_LOGS:
            logs = {}
        super().__init__(api_key=api_key, cache=cache, logs=logs)

        self.model_name: str = (
            model_name if model_name is not None else self.default_model_name
        )
        self.output_max_length: int = (
            output_max_length or self.default_output_max_length
        )
        self.temperature: float = (
            temperature
            if temperature is not None
            else self.default_temperature
        )

        self.prompt_template: str = (
            prompt_template or self.default_prompt_template
        )
        self.structured_output: Optional[Type[BaseModel]] = structured_output
        if api_params is DEFAULT_API_PARAMS:
            api_params = deepcopy(self.default_api_params or {})

        self.system_message = system_message
        self.api_params = api_params

        if device not in ['cpu', 'cuda', 'auto']:
            raise Exception(
                f'Device {device} not supported. Options: cpu, cuda, auto.'
            )

        cuda_available = torch.cuda.is_available()
        self.device_name: str = (
            'cpu' if device == 'cpu' or not cuda_available else 'cuda'
        )
        self.device = torch.device(self.device_name)

        self._validate()
        self._setup()

    def _validate(self) -> None:
        """Raise an error if the initial parameters are not valid."""
        return

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        return

    @abstractmethod
    def generate(
        self,
        query: str,
        context: list[str],
    ) -> str | BaseModel:
        """Generate text with optional language parameter.

        Parameters
        ----------
        query : str
            The input query or prompt.
        context : list[str]
            Additional context information for the generation.

        Returns
        -------
        str
            Generated text based on query and context.
        """
        ...
