"""Hugging Face classes for text generation."""

from __future__ import annotations

from transformers import T5ForConditionalGeneration, T5Tokenizer

from rago.generation.base import GenerationBase


class HuggingFaceGen(GenerationBase):
    """HuggingFaceGen."""

    def __init__(
        self, model_name: str = 't5-small', output_max_length: int = 500
    ) -> None:
        """Initialize HuggingFaceGen."""
        if model_name == 't5-small':
            self._set_t5_small_models()
        else:
            raise Exception(f'The given model {model_name} is not supported.')

        self.output_max_length = output_max_length

    def _set_t5_small_models(self) -> None:
        """Set models to t5-small models."""
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def generate(self, query: str, context: list[str]) -> str:
        """Generate the text from the query and augmented context."""
        # Prepare the input for the generative model
        input_text = f"Question: {query} Context: {' '.join(context)}"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')

        # Generate the response
        outputs = self.model.generate(
            input_ids, max_length=self.output_max_length
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return str(response)
