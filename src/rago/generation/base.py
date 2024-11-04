"""Base classes for generation."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch

from typeguard import typechecked


@typechecked
class GenerationBase:
    """Generic Generation class."""

    api_key: str = ''
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

    # default parameters that can be overwritten by the derived class
    default_device_name: str = 'cpu'
    default_model_name: str = ''
    default_temperature: float = 0.5
    default_output_max_length: int = 500
    default_prompt_template: str = (
        'question: \n```\n{query}\n```\ncontext: ```\n{context}\n```'
    )

    def __init__(
        self,
        model_name: str = '',
        api_key: str = '',
        temperature: float = 0.5,
        prompt_template: str = '',
        output_max_length: int = 500,
        device: str = 'auto',
    ) -> None:
        """Initialize Generation class.

        Parameters
        ----------
        model_name : str
            The name of the model to use.
        api_key : str
        temperature : float
        prompt_template: str
        output_max_length : int
            Maximum length of the generated output.
        device: str (default=auto)
        """
        self.api_key = api_key
        self.model_name = model_name or self.default_model_name
        self.output_max_length = (
            output_max_length or self.default_output_max_length
        )
        self.temperature = temperature or self.default_temperature

        self.prompt_template = prompt_template or self.default_prompt_template

        if self.device_name not in ['cpu', 'cuda', 'auto']:
            raise Exception(
                f'Device {self.device_name} not supported. '
                'Options: cpu, cuda, auto.'
            )

        cuda_available = torch.cuda.is_available()
        self.device_name = (
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
    ) -> str:
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
