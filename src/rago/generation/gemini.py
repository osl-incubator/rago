"""GeminiGen class for text generation using Google's Gemini model."""

from __future__ import annotations

from typing import cast

import google.generativeai as genai

from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class GeminiGen(GenerationBase):
    """Gemini generation model for text generation."""

    default_model_name: str = 'gemini-1.5-flash'

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def generate(self, query: str, context: list[str]) -> str:
        """Generate text using Gemini model support."""
        input_text = self.prompt_template.format(
            query=query, context=' '.join(context)
        )

        response = self.model.generate_content(input_text)
        return cast(str, response.text.strip())
