# src/rago/generation/llama3.py

"""Llama 3.2 1B classes for text generation."""

from __future__ import annotations

from typing import cast

import torch

from langdetect import detect
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class LlamaGen(GenerationBase):
    """Llama Generation class."""

    def __init__(
        self,
        model_name: str = 'meta-llama/Llama-3.2-1B',
        apikey: str = '',
        temperature: float = 0.5,
        output_max_length: int = 500,
        device: str = 'auto',
    ) -> None:
        """Initialize LlamaGen."""
        if not model_name.startswith('meta-llama/'):
            raise Exception(
                f'The given model name {model_name} is not provided by meta.'
            )

        super().__init__(
            model_name=model_name,
            apikey=apikey,
            temperature=temperature,
            output_max_length=output_max_length,
            device=device,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=apikey
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=apikey,
            torch_dtype=torch.float16
            if self.device_name == 'cuda'
            else torch.float32,
        )

        self.generator = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device_name == 'cuda' else -1,
        )

    def generate(self, query: str, context: list[str]) -> str:
        """Generate text using Llama model with language support."""
        input_text = f"Question: {query} Context: {' '.join(context)}"
        language = str(detect(query)) or 'en'

        self.tokenizer.lang_code = language

        response = self.generator(
            input_text,
            max_length=self.output_max_length,
            do_sample=True,
            temperature=self.temperature,
            num_return_sequences=1,
        )

        breakpoint()

        return cast(str, response[0].get('generated_text', ''))
