# src/rago/generation/llama3.py

"""Llama 3.2 1B classes for text generation."""

from __future__ import annotations

from typing import List, cast

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from rago.generation.base import GenerationBase


class LlamaV32M1BGen(GenerationBase):
    """Llama 3.2 1B Generation class."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    generator: pipeline

    def __init__(
        self,
        model_name: str = 'meta-llama/Llama-3.2-1B',
        output_max_length: int = 500,
        apikey: str = '',
    ) -> None:
        """Initialize LlamaV32M1BGen."""
        super().__init__(
            model_name=model_name, output_max_length=output_max_length
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=apikey
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, token=apikey
        )
        self.generator = pipeline(
            'text-generation', model=self.model, tokenizer=self.tokenizer
        )

    def generate(self, query: str, context: List[str]) -> str:
        """Generate text using Llama 3.2 1B model."""
        input_text = f"Question: {query} Context: {' '.join(context)}"
        response = self.generator(
            input_text,
            max_length=self.output_max_length,
            do_sample=True,
            temperature=0.7,
        )
        return cast(str, response[0].get('generated_text', ''))
