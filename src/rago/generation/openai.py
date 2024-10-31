"""OpenAI Generation Model class for flexible GPT-based text generation."""

from __future__ import annotations

from typing import cast

import openai

from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class OpenAIGen(GenerationBase):
    """OpenAI generation model for text generation."""

    default_model_name = 'gpt-3.5-turbo'

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        self.model = openai.OpenAI(api_key=self.api_key)

    def generate(
        self,
        query: str,
        context: list[str],
    ) -> str:
        """Generate text using OpenAI's API with dynamic model support."""
        input_text = self.prompt_template.format(
            query=query, context=' '.join(context)
        )

        if not self.model:
            raise Exception('The model was not created.')

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': input_text}],
            max_tokens=self.output_max_length,
            temperature=self.temperature,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )

        return cast(str, response.choices[0].message.content.strip())
