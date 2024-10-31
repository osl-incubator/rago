"""OpenAI Generation Model class for flexible GPT-based text generation."""

from __future__ import annotations

from typing import cast

import openai

from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class OpenAIGPTGen(GenerationBase):
    """OpenAI generation model for text generation."""

    def __init__(
        self,
        model_name: str = 'gpt-4',
        output_max_tokens: int = 500,
        api_key: str = '',
    ) -> None:
        """Initialize OpenAIGenerationModel with OpenAI's model."""
        super().__init__(
            model_name=model_name, output_max_length=output_max_tokens
        )
        openai.api_key = api_key

    def generate(
        self,
        query: str,
        context: list[str],
        language: str = 'en',
    ) -> str:
        """Generate text using OpenAI's API with dynamic model support."""
        input_text = (
            f"Question: {query}\nContext: {' '.join(context)}\n"
            f"Answer in {language}:"
        )

        response = openai.Completion.create(  # type: ignore[no-untyped-call]
            model=self.model_name,
            messages=[{'role': 'user', 'content': input_text}],
            max_tokens=self.output_max_length,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )

        return cast(str, response['choices'][0]['message']['content'].strip())
