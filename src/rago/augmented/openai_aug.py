"""OpenAIAug class for query augmentation using OpenAI API."""

from __future__ import annotations

import openai

from typeguard import typechecked

from rago.augmented.base import AugmentedBase


@typechecked
class OpenAIAug(AugmentedBase):
    """OpenAIAug class for query augmentation using OpenAI API."""

    def __init__(self, model_name: str = 'gpt-4', k: int = 1) -> None:
        """Initialize the OpenAIAug class."""
        self.model_name = model_name
        self.k = k

    def search(
        self, query: str, documents: list[str], k: int = 1
    ) -> list[str]:
        """Augment the query by expanding or rephrasing it using OpenAI."""
        prompt = f"Retrieval: '{query}'\nContext: {' '.join(documents)}"

        response = openai.Completion.create(  # type: ignore[attr-defined]
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=50,
            temperature=0.5,
        )

        augmented_query = response.choices[0]['message']['content'].strip()
        return [augmented_query] * self.k
