"""Mistral AI Generation Model class for text generation."""

from __future__ import annotations

from typing import cast

import requests
from pydantic import BaseModel
from typeguard import typechecked

from rago.generation.base import GenerationBase

@typechecked
class MistralGen(GenerationBase):
    """Mistral AI generation model for text generation."""

    default_model_name: str = 'mistral-large'
    api_url: str = 'https://api.mistral.ai/v1/chat/completions'
    default_temperature: float = 0.7
    default_output_max_length: int = 500
    default_api_params = {
        'top_p': 0.9,
        'max_tokens': 500,
    }

    def _setup(self) -> None:
        """Ensure API key is set."""
        if not self.api_key:
            raise RuntimeError("API key is required for Mistral AI.")

    def generate(self, query: str, context: list[str]) -> str | BaseModel:
        """Generate text using Mistral AI's API."""
        input_text = self.prompt_template.format(
            query=query, context=' '.join(context)
        )

        api_params = self.api_params if self.api_params else self.default_api_params
        model_params = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': input_text}],
            'temperature': self.temperature,
            **api_params,
        }

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        response = requests.post(self.api_url, json=model_params, headers=headers)
        response.raise_for_status()
        result = response.json()

        self.logs['generation'] = {
            'model': self.model_name,
            'input_text': input_text,
            'parameters': model_params,
        }

        if self.structured_output:
            return cast(BaseModel, result)

        return cast(str, result['choices'][0]['message']['content'].strip())
