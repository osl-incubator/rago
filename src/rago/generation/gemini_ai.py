"""GeminiAIGen class for text generation using Google's Gemini model."""

from __future__ import annotations

from typing import cast

import google.generativeai as genai

from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class GeminiAIGen(GenerationBase):
    """Gemini generation model for text generation."""

    def __init__(
        self,
        model_name: str = 'gemini-1.5-flash',
        output_max_length: int = 500,
        api_key: str = '',
    ) -> None:
        """Initialize GeminiAIGen class with Google's Gemini model."""
        super().__init__(
            model_name=model_name, output_max_length=output_max_length
        )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(
        self, query: str, context: list[str], language: str = 'en'
    ) -> str:
        """
        Generate text using Google's Generative AI with Gemini model support.

        Returns
        -------
        str
            The generated response.
        """
        input_text = (
            f"Question: {query}\nContext: {' '.join(context)}\n"
            f"Answer in {language}:"
        )

        response = self.model.generate_content(input_text)  # Simplified call
        return cast(str, response['text'].strip())
