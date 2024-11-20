"""Hugging Face classes for text generation."""

from __future__ import annotations

import warnings

import torch

from transformers import T5ForConditionalGeneration, T5Tokenizer
from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class HuggingFaceGen(GenerationBase):
    """HuggingFaceGen."""

    default_model_name = 't5-small'

    def _validate(self) -> None:
        if self.model_name != 't5-small':
            raise Exception(
                f'The given model {self.model_name} is not supported.'
            )

        if self.structured_output:
            warnings.warn(
                'Structured output is not supported yet in '
                f'{self.__class__.__name__}.'
            )

    def _setup(self) -> None:
        """Set models to t5-small models."""
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.model = model.to(self.device)

    def generate(self, query: str, context: list[str]) -> str:
        """Generate the text from the query and augmented context."""
        with torch.no_grad():
            input_text = self.prompt_template.format(
                query=query, context=' '.join(context)
            )
            input_ids = self.tokenizer.encode(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
            ).to(self.device_name)

            model_params = dict(
                inputs=input_ids,
                max_length=self.output_max_length,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            outputs = self.model.generate(**model_params)

            self.logs['model_params'] = model_params

            response = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

        if self.device_name == 'cuda':
            torch.cuda.empty_cache()

        return str(response)
