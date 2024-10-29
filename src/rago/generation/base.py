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
    tokenizer: Any
    temperature: float = 0.5
    output_max_length: int = 500

    @abstractmethod
    def __init__(
        self,
        model_name: str = '',
        api_key: str = '',
        temperature: float = 0.5,
        output_max_length: int = 500,
        device: str = 'auto',
    ) -> None:
        """Initialize GenerationBase.

        Parameters
        ----------
        model_name : str
            The name of the model to use.
        api_key : str
        temperature : float
        output_max_length : int
            Maximum length of the generated output.
        device: str (default=auto)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.output_max_length = output_max_length
        self.temperature = temperature

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
