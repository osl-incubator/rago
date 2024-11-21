"""OpenAI Generation Model class for flexible GPT-based text generation."""

from __future__ import annotations

from typing import cast

import instructor
import openai

from pydantic import BaseModel
from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class OpenAIGen(GenerationBase):
    """OpenAI generation model for text generation."""

    default_model_name = 'gpt-3.5-turbo'

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        model = openai.OpenAI(api_key=self.api_key)

        self.model = (
            instructor.from_openai(model) if self.structured_output else model
        )

    def generate(
        self,
        query: str,
        context: list[str],
    ) -> str | BaseModel:
        """Generate text using OpenAI's API with dynamic model support."""
        input_text = self.prompt_template.format(
            query=query, context=' '.join(context)
        )

        if not self.model:
            raise Exception('The model was not created.')

        model_params = dict(
            model=self.model_name,
            messages=[{'role': 'user', 'content': input_text}],
            max_tokens=self.output_max_length,
            temperature=self.temperature,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )

        if self.structured_output:
            model_params['response_model'] = self.structured_output

        response = self.model.chat.completions.create(**model_params)

        self.logs['model_params'] = model_params

        has_choices = hasattr(response, 'choices')

        if has_choices and isinstance(response.choices, list):
            return cast(str, response.choices[0].message.content.strip())
        return cast(BaseModel, response)
