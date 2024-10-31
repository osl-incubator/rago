"""OpenAI Generation Model class for flexible GPT-based text generation."""

from __future__ import annotations

from typing import cast

import openai

from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class OpenAIGPTGen(GenerationBase):
    """OpenAI generation model for text generation."""

    default_model_name = 'gpt-4'

    def generate(
        self,
        query: str,
        context: list[str],
    ) -> str:
        """Generate text using OpenAI's API with dynamic model support."""
        input_text = self.prompt_template.format(
            query=query, context=' '.join(context)
        )

        response = openai.Completion.create(  # type: ignore[no-untyped-call]
            model=self.model_name,
            messages=[{'role': 'user', 'content': input_text}],
            max_tokens=self.output_max_length,
            temperature=self.temperature,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )

        return cast(str, response['choices'][0]['message']['content'].strip())
