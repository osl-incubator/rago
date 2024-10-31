"""OpenAIAug class for query augmentation using OpenAI API."""

from __future__ import annotations

from typing import cast

import openai

from typeguard import typechecked

from rago.augmented.base import AugmentedBase


@typechecked
class OpenAIAug(AugmentedBase):
    """OpenAIAug class for query augmentation using OpenAI API."""

    default_model_name = 'gpt-4'
    default_k = 2
    default_result_separator = '\n'

    def search(
        self, query: str, documents: list[str], k: int = 0
    ) -> list[str]:
        """Augment the query by expanding or rephrasing it using OpenAI."""
        k = k or self.k
        prompt = self.prompt_template.format(
            query=query, context=' '.join(documents), k=k
        )

        response = openai.Completion.create(  # type: ignore[no-untyped-call]
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=self.output_max_length,
            temperature=self.temperature,
        )

        augmented_query = cast(
            str, response.choices[0]['message']['content'].strip()
        )
        return augmented_query.split(self.result_separator)[:k]
