"""Llama generation module."""

from __future__ import annotations

import warnings

import torch

from langdetect import detect
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class LlamaGen(GenerationBase):
    """Llama Generation class."""

    default_model_name: str = 'meta-llama/Llama-3.2-1B'
    default_temperature: float = 0.5
    default_output_max_length: int = 500

    def _validate(self) -> None:
        """Raise an error if the initial parameters are not valid."""
        if not self.model_name.startswith('meta-llama/'):
            raise Exception(
                f'The given model name {self.model_name} is not provided '
                'by meta.'
            )

        if self.structured_output:
            warnings.warn(
                'Structured output is not supported yet in '
                f'{self.__class__.__name__}.'
            )

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=self.api_key
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.api_key,
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

        model_params = dict(
            text_inputs=input_text,
            max_new_tokens=self.output_max_length,
            do_sample=True,
            temperature=self.temperature,
            top_k=50,  # todo: check if it is necessary
            top_p=0.95,  # todo: check if it is necessary
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response = self.generator(**model_params)

        self.logs['model_params'] = model_params

        # Extract and return the answer only
        answer = str(response[0].get('generated_text', ''))
        # Strip off any redundant text after the answer itself
        return answer.split('Answer:')[-1].strip()
