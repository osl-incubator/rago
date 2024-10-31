"""GeminiAug class for query augmentation using Google's Gemini Model."""

from __future__ import annotations

import google.generativeai as genai

from typeguard import typechecked

from rago.augmented.base import AugmentedBase


@typechecked
class GeminiAug(AugmentedBase):
    """GeminiAug class for query augmentation using Gemini API."""

    def __init__(
        self,
        model_name: str = 'gemini-1.5-flash',
        k: int = 1,
        api_key: str = '',
    ) -> None:
        """Initialize the GeminiAug class."""
        self.model_name = model_name
        self.k = k
        genai.configure(api_key=api_key)

    def search(
        self, query: str, documents: list[str], k: int = 1
    ) -> list[str]:
        """Augment the query by expanding or rephrasing it using Gemini."""
        prompt = f"Retrieval: '{query}'\nContext: {' '.join(documents)}"

        response = genai.GenerativeModel(self.model_name).generate_content(
            prompt
        )

        augmented_query = (
            response.text.strip()
            if hasattr(response, 'text')
            else response[0].text.strip()
        )
        return [augmented_query] * self.k
