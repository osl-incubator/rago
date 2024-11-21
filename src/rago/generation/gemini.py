"""GeminiGen class for text generation using Google's Gemini model."""

from __future__ import annotations

from typing import cast

import google.generativeai as genai
import instructor

from pydantic import BaseModel
from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class GeminiGen(GenerationBase):
    """Gemini generation model for text generation."""

    default_model_name: str = 'gemini-1.5-flash'

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        self.model = (
            instructor.from_gemini(
                client=model,
                mode=instructor.Mode.GEMINI_JSON,
            )
            if self.structured_output
            else model
        )

    def generate(self, query: str, context: list[str]) -> str | BaseModel:
        """Generate text using Gemini model support."""
        input_text = self.prompt_template.format(
            query=query, context=' '.join(context)
        )

        if not self.structured_output:
            models_params_gen = {'contents': input_text}
            response = self.model.generate_content(**models_params_gen)
            self.logs['model_params'] = models_params_gen
            return cast(str, response.text.strip())

        messages = [
            {'role': 'user', 'content': input_text},
        ]
        model_params = {
            'messages': messages,
            'response_model': self.structured_output,
        }

        response = self.model.create(
            **model_params,
        )

        self.logs['model_params'] = model_params

        return cast(BaseModel, response)
