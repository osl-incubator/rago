"""Groq class for text generation."""

from __future__ import annotations

from typing import cast

import instructor
import openai

from pydantic import BaseModel
from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class GroqGen(GenerationBase):
    """Groq generation model for text generation."""

    default_model_name = 'gemma2-9b-it'
    default_api_params = {
        'top_p': 1.0,
    }

    def _setup(self) -> None:
        """Set up the Groq client."""
        groq_api_key = self.api_key
        if not groq_api_key:
            raise Exception('GROQ_API_KEY environment variable is not set')

        # Can use Groq client as well.
        groq_client = openai.OpenAI(
            base_url='https://api.groq.com/openai/v1', api_key=groq_api_key
        )

        # Optionally use instructor if structured output is needed
        self.model = (
            instructor.from_openai(groq_client)
            if self.structured_output
            else groq_client
        )

    def generate(
        self,
        query: str,
        context: list[str],
    ) -> str | BaseModel:
        """Generate text using the Groq AP."""
        input_text = self.prompt_template.format(
            query=query, context=' '.join(context)
        )

        if not self.model:
            raise Exception('The model was not created.')

        api_params = (
            self.api_params if self.api_params else self.default_api_params
        )

        messages = []
        if self.system_message:
            messages.append({'role': 'system', 'content': self.system_message})
        messages.append({'role': 'user', 'content': input_text})

        model_params = dict(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=self.output_max_length,
            temperature=self.temperature,
            **api_params,
        )

        if self.structured_output:
            model_params['response_model'] = self.structured_output

        response = self.model.chat.completions.create(**model_params)
        self.logs['model_params'] = model_params

        if hasattr(response, 'choices') and isinstance(response.choices, list):
            return cast(str, response.choices[0].message.content.strip())

        return cast(BaseModel, response)
