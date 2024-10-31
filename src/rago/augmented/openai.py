"""OpenAIAug class for query augmentation using OpenAI API."""

from __future__ import annotations

from typing import cast

import openai

from typeguard import typechecked

from rago.augmented.base import AugmentedBase


@typechecked
class OpenAIAug(AugmentedBase):
    """OpenAIAug class for query augmentation using OpenAI API."""

    default_model_name = 'gpt-3.5-turbo'
    default_k = 2
    default_result_separator = '\n'

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        self.model = openai.OpenAI(api_key=self.api_key)

    def search(
        self, query: str, documents: list[str], k: int = 0
    ) -> list[str]:
        """Augment the query by expanding or rephrasing it using OpenAI."""
        k = k or self.k
        prompt = self.prompt_template.format(
            query=query, context=' '.join(documents), k=k
        )

        if not self.model:
            raise Exception('The model was not created.')

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=self.output_max_length,
            temperature=self.temperature,
        )

        augmented_query = cast(
            str, response.choices[0].message.content.strip()
        )
        return augmented_query.split(self.result_separator)[:k]
