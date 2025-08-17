"""Phi generation module."""

from __future__ import annotations

import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class PhiGen(GenerationBase):
    """Phi Generation class."""

    default_model_name: str = 'microsoft/phi-2'
    default_temperature: float = 0.7
    default_output_max_length: int = 500
    default_api_params = {
        'top_p': 0.9,
        'num_return_sequences': 1,
    }

    def _validate(self) -> None:
        """Raise an error if the initial parameters are not valid."""
        if not self.model_name.startswith('microsoft/phi-'):
            raise Exception(
                f'The given model name {self.model_name} is not a Phi model '
                'from Microsoft.'
            )

        if self.structured_output:
            warnings.warn(
                'Structured output is not supported yet in '
                f'{self.__class__.__name__}.'
            )

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore
            self.model_name, trust_remote_code=True
        )

        device_map = 'auto' if self.device_name == 'cuda' else None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype='auto',
            device_map=device_map,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.generation_config = GenerationConfig.from_pretrained(
            self.model_name
        )
        self.model.generation_config.pad_token_id = (
            self.model.generation_config.eos_token_id
        )

    def generate(self, query: str, context: list[str]) -> str:
        """Generate text using Phi model with context."""
        full_prompt = f'{query}\nContext: {" ".join(context)}'

        inputs = self.tokenizer(
            full_prompt, return_tensors='pt', return_attention_mask=True
        ).to(self.model.device)

        model_params = dict(
            max_new_tokens=self.output_max_length,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.default_api_params['top_p'],
            num_return_sequences=self.default_api_params[
                'num_return_sequences'
            ],
        )

        self.logs['model_params'] = model_params

        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **model_params,
        )

        answer: str = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        return answer.strip()
