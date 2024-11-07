"""GeminiAug class for query augmentation using Google's Gemini Model."""

from __future__ import annotations

import google.generativeai as genai

from typeguard import typechecked

from rago.augmented.base import AugmentedBase


@typechecked
class GeminiAug(AugmentedBase):
    """GeminiAug class for query augmentation using Gemini API."""

    default_model_name: str = 'gemini-1.5-flash'
    default_top_k: int = 1

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        genai.configure(api_key=self.api_key)

    def search(
        self, query: str, documents: list[str], top_k: int = 0
    ) -> list[str]:
        """Augment the query by expanding or rephrasing it using Gemini."""
        top_k = top_k or self.top_k
        prompt = self.prompt_template.format(
            query=query, context=' '.join(documents), top_k=top_k
        )

        response = genai.GenerativeModel(self.model_name).generate_content(
            prompt
        )

        augmented_query = str(
            response.text.strip()
            if hasattr(response, 'text')
            else response[0].text.strip()
        )
        return augmented_query.split(self.result_separator)[:top_k]
