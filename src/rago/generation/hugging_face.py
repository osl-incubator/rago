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
        api_key: str = '',
        temperature: float = 0.5,
        output_max_length: int = 500,
        device: str = 'auto',
    ) -> None:
        """Initialize HuggingFaceGen."""
        if model_name != 't5-small':
            raise Exception(f'The given model {model_name} is not supported.')

        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            output_max_length=output_max_length,
            device=device,
        )

        self._set_t5_small_models()

    def _set_t5_small_models(self) -> None:
        """Set models to t5-small models."""
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.model = self.model.to(self.device)

    def generate(self, query: str, context: list[str]) -> str:
        """Generate the text from the query and augmented context."""
        with torch.no_grad():
            input_text = f"Question: {query} Context: {' '.join(context)}"
            input_ids = self.tokenizer.encode(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
            ).to(self.device_name)

            outputs = self.model.generate(
                input_ids,
                max_length=self.output_max_length,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            response = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

        if self.device_name == 'cuda':
            torch.cuda.empty_cache()

        return str(response)
