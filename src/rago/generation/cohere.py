"""CohereGen class for text generation using Cohere's API."""

from __future__ import annotations

import json

from typing import cast

import cohere
import instructor

from pydantic import BaseModel
from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class CohereGen(GenerationBase):
    """Cohere generation model for text generation."""

    default_model_name: str = 'command-r-plus-08-2024'
    default_api_params = {
        'p': 0.9,
    }

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        model = cohere.ClientV2(api_key=self.api_key)
        self.model = (
            instructor.from_cohere(
                client=model,
                mode=instructor.Mode.COHERE_JSON_SCHEMA,
                model_name=self.model_name,
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
            # Explicit instruction to generate JSON output.
            system_instruction = (
                'Generate a JSON object that strictly follows the provided  '
                'JSON schema. Do not include any additional text.'
            )
            if self.system_message:
                system_instruction += ' ' + self.system_message
            messages.append({'role': 'system', 'content': system_instruction})
            messages.append({'role': 'user', 'content': input_text})

            response_format_config = {
                'type': 'json_object',
                'json_schema': (
                    self.structured_output
                    if isinstance(self.structured_output, dict)
                    else self.structured_output.model_json_schema()
                ),
            }
            model_params = {
                'messages': messages,
                'max_tokens': self.output_max_length,
                'temperature': self.temperature,
                'model': self.model_name,
                'response_format': response_format_config,
                **api_params,
            }

            response = self.model.client.chat(**model_params)
            self.logs['model_params'] = model_params
            json_text = response.message.content[0].text
            parsed_dict = json.loads(json_text)
            parsed_model = self.structured_output(**parsed_dict)
            return parsed_model

        if self.system_message:
            messages = [
                {'role': 'system', 'content': self.system_message},
                {'role': 'user', 'content': input_text},
            ]
            model_params = {
                'model': self.model_name,
                'messages': messages,
                'max_tokens': self.output_max_length,
                'temperature': self.temperature,
                **api_params,
            }
            response = self.model.chat(**model_params)
            self.logs['model_params'] = model_params
            return cast(str, response.text)

        model_params = {
            'model': self.model_name,
            'prompt': input_text,
            'max_tokens': self.output_max_length,
            'temperature': self.temperature,
            **api_params,
        }
        response = self.model.generate(**model_params)
        self.logs['model_params'] = model_params
        return cast(str, response.generations[0].text.strip())
