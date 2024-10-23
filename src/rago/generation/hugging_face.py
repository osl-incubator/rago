"""Hugging Face classes for text generation."""

from __future__ import annotations

import torch

from transformers import T5ForConditionalGeneration, T5Tokenizer
from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class HuggingFaceGen(GenerationBase):
    """HuggingFaceGen."""

    def __init__(
        self,
        model_name: str = 't5-small',
        output_max_length: int = 500,
        device: str = 'auto',
    ) -> None:
        """
        Initialize HuggingFaceGen.

        Parameters
        ----------
        model_name : str
            The name of the Hugging Face model to use.
        output_max_length : int
            The maximum length for the generated output.
        device : str (default 'auto')
            Device for running the model ('cpu', 'cuda', or 'auto'),
            default is 'auto'.
        """
        if model_name == 't5-small':
            self._set_t5_small_models(device)
        else:
            raise Exception(f'The given model {model_name} is not supported.')

        self.output_max_length = output_max_length
        self.device = device

    def _set_t5_small_models(self, device: str) -> None:
        """Set models to t5-small models."""
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

        # Move the model to the appropriate device (cpu/cuda/auto)
        if device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.to('cuda')
        elif device == 'cpu':
            self.model = self.model.to('cpu')
        elif device == 'auto':
            self.model = self.model.to(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )

    def generate(
        self, query: str, context: list[str], device: str = 'auto'
    ) -> str:
        """
        Generate the text from the query and augmented context.

        Parameters
        ----------
        query : str
            The query or prompt from the user.
        context : list[str]
            Contextual information for the query.
        device : str, optional
            Device for generation (e.g., 'auto', 'cpu', 'cuda'),
            by default 'auto'.

        Returns
        -------
        str
            The generated response.
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = self.model.to(device)

        with torch.no_grad():
            input_text = f"Question: {query} Context: {' '.join(context)}"
            input_ids = self.tokenizer.encode(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
            ).to(device)

            outputs = self.model.generate(
                input_ids,
                max_length=self.output_max_length,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            response = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

        torch.cuda.empty_cache()

        return str(response)
