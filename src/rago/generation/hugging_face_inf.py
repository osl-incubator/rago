"""Hugging Face inferencing classes for text generation."""

from __future__ import annotations

from huggingface_hub import InferenceClient
from pydantic import BaseModel
from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class HuggingFaceInfGen(GenerationBase):
    """HuggingFaceGen with InferenceClient."""

    default_model_name: str = 'google/gemma-2-2b-it'
    default_api_params = {
        'temperature': 0.7,
        'max_new_tokens': 512,
    }

    def _setup(self) -> None:
        """Set up InferenceClient with API key."""
        self.client = InferenceClient(
            provider='hf-inference', api_key=self.api_key
        )

    def generate(self, query: str, context: list[str]) -> str | BaseModel:
        """Generate the text from the query and augmented context."""
        input_text = self.prompt_template.format(
            query=query, context=' '.join(context)
        )
        if self.system_message:
            input_text = f'{self.system_message}\n{input_text}'

        api_params = self.api_params or self.default_api_params

        self.logs['model_params'] = {
            'model': self.model_name,
            'inputs': input_text,
            'parameters': api_params,
        }
        generated_text = self.client.text_generation(
            prompt=input_text,
            model=self.model_name,
            max_new_tokens=api_params['max_new_tokens'],
            temperature=api_params['temperature'],
        )

        return generated_text.strip()
