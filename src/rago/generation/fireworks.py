"""FireworksGen class for text generation using Fireworks API."""

from __future__ import annotations

from typing import cast

import instructor

from fireworks.client import Fireworks
from pydantic import BaseModel
from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class FireworksGen(GenerationBase):
    """Fireworks AI generation model for text generation."""

    default_model_name: str = 'accounts/fireworks/models/llama-v3-8b-instruct'
    default_api_params = {
        'top_p': 0.9,
    }

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        model = Fireworks(api_key=self.api_key)

        self.model = (
            instructor.from_fireworks(
                client=model,
                mode=instructor.Mode.FIREWORKS_JSON,
            )
            if self.structured_output
            else model
        )

    def generate(self, query: str, context: list[str]) -> str | BaseModel:
        """Generate text using Fireworks AI's API."""
        input_text = self.prompt_template.format(
            query=query, context=' '.join(context)
        )

        api_params = self.api_params or self.default_api_params

        messages = []
        if self.system_message:
            messages.append({'role': 'system', 'content': self.system_message})
        messages.append({'role': 'user', 'content': input_text})

        model_params = {
            'model': self.model_name,
            'messages': messages,
            'max_tokens': self.output_max_length,
            'temperature': self.temperature,
            **api_params,
        }

        if self.structured_output:
            model_params['response_model'] = self.structured_output
            response = self.model.chat.completions.create(**model_params)
            self.logs['model_params'] = model_params
            return cast(BaseModel, response)

        response = self.model.chat.completions.create(**model_params)
        self.logs['model_params'] = model_params
        return cast(str, response.choices[0].message.content.strip())
