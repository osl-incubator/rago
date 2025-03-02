"""CohereGen class for text generation using Cohere's API."""

from __future__ import annotations

from typing import cast

import cohere
import instructor

from pydantic import BaseModel
from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class CohereGen(GenerationBase):
    """Cohere generation model for text generation."""

    default_model_name: str = 'command'
    default_api_params = {
        'p': 0.9,
    }

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        model = cohere.Client(api_key=self.api_key)

        self.model = (
            instructor.from_cohere(
                client=model,
                mode=instructor.Mode.COHERE_JSON_SCHEMA,
                model_name = self.model_name
            )
            if self.structured_output
            else model
        )

    def generate(self, query: str, context: list[str]) -> str | BaseModel:
        """Generate text using Cohere's API."""
        input_text = self.prompt_template.format(
            query=query, context=' '.join(context)
        )

        api_params = self.api_params or self.default_api_params

        if self.structured_output:
            messages = []
            if self.system_message:
                messages.append({"role": "SYSTEM", "message": self.system_message})
            messages.append({"role": "USER", "message": input_text})

            model_params = {
                "chat_history": messages,
                "max_tokens": self.output_max_length,
                "temperature": self.temperature,
                "response_model": self.structured_output,
                **api_params,
            }

            response = self.model.chat(**model_params)
            self.logs['model_params'] = model_params
            return cast(BaseModel, response)

        if self.system_message:
            messages = [
                {"role": "SYSTEM", "message": self.system_message},
                {"role": "USER", "message": input_text}
            ]

            model_params = {
                "model": self.model_name,
                "chat_history": messages,
                "max_tokens": self.output_max_length,
                "temperature": self.temperature,
                **api_params,
            }

            response = self.model.chat(**model_params)
            self.logs['model_params'] = model_params
            return cast(str, response.text)

        # Use generate for simple completions
        model_params = {
            "model": self.model_name,
            "prompt": input_text,
            "max_tokens": self.output_max_length,
            "temperature": self.temperature,
            **api_params,
        }

        response = self.model.generate(**model_params)
        self.logs['model_params'] = model_params
        return cast(str, response.generations[0].text.strip())
