"""Hugging Face classes for text generation with optimization."""

from __future__ import annotations

import warnings
import torch

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    BitsAndBytesConfig
)
from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class HuggingFaceGen(GenerationBase):
    """Optimized HuggingFaceGen with 4-bit quantization."""

    default_model_name = 't5-small'

    def _validate(self) -> None:
        """Validate if the model parameters are correct."""
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
        """Set up the model with optimized quantization."""

        # Define quantization configuration for efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,   # Enable 4-bit quantization
            bnb_4bit_compute_dtype=torch.float16,  # Use FP16 for better performance
            bnb_4bit_use_double_quant=True,  # Enable double quantization
        )

        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)

        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,  # Apply quantization
            device_map="auto"  # Automatically place on the best available device
        )

    def generate(self, query: str, context: list[str]) -> str:
        """Generate text from query and context using the optimized model."""
        with torch.no_grad():
            input_text = self.prompt_template.format(
                query=query, context=' '.join(context)
            )
            input_ids = self.tokenizer.encode(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
            ).to(self.model.device)  # Ensure tensor is on the correct device

            api_params = (
                self.api_params if self.api_params else self.default_api_params
            )

            model_params = dict(
                inputs=input_ids,
                max_length=self.output_max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                **api_params,
            )

            outputs = self.model.generate(**model_params)

            self.logs['model_params'] = model_params

            response = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

        # Clear CUDA cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return str(response)
