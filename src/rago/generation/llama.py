"""Llama generation module."""

from __future__ import annotations

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
        api_key: str = '',
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
            api_key=api_key,
            temperature=temperature,
            output_max_length=output_max_length,
            device=device,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=api_key
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=api_key,
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
        input_text = self.prompt_template.format(
            query=query, context=' '.join(context)
        )

        # Detect and set the language code for multilingual models (optional)
        language = str(detect(query)) or 'en'
        self.tokenizer.lang_code = language

        # Generate the response with adjusted parameters
        response = self.generator(
            input_text,
            max_new_tokens=self.output_max_length,
            do_sample=True,
            temperature=self.temperature,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Extract and return the answer only
        answer = str(response[0].get('generated_text', ''))
        # Strip off any redundant text after the answer itself
        return answer.split('Answer:')[-1].strip()
