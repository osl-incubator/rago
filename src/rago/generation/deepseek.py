"""DeepSeek generation module."""

from __future__ import annotations

import warnings

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class DeepSeekGen(GenerationBase):
    """DeepSeek Generation class."""

    default_model_name: str = 'deepseek-ai/deepseek-llm-7b-chat'
    default_temperature: float = 0.5
    default_output_max_length: int = 500
    default_api_params = {
        'top_p': 0.9,
        'num_return_sequences': 1,
    }

    def _validate(self) -> None:
        """Raise an error if the initial parameters are not valid."""
        if not self.model_name.startswith('deepseek-ai/'):
            raise Exception(
                f'The given model name {self.model_name} is not provided '
                'by DeepSeek.'
            )

        if self.structured_output:
            warnings.warn(
                'Structured output is not supported yet in '
                f'{self.__class__.__name__}.'
            )

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore
            self.model_name
        )

        device_map = 'auto' if self.device_name == 'cuda' else None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16
            if self.device_name == 'cuda'
            else torch.float32,
            device_map=device_map,
        )

        self.model.generation_config = GenerationConfig.from_pretrained(
            self.model_name
        )
        self.model.generation_config.pad_token_id = (
            self.model.generation_config.eos_token_id
        )

    def generate(self, query: str, context: list[str]) -> str:
        """Generate text using DeepSeek model with chat template."""
        messages = [
            {
                'role': 'user',
                'content': f'{query}\nContext: {" ".join(context)}',
            }
        ]

        input_tensor = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors='pt'
        ).to(self.model.device)

        model_params = dict(
            max_new_tokens=self.output_max_length,
            do_sample=True,
            temperature=self.temperature,
        )

        self.logs['model_params'] = model_params

        outputs = self.model.generate(input_tensor, **model_params)

        answer: str = str(
            self.tokenizer.decode(
                outputs[0][input_tensor.shape[1] :], skip_special_tokens=True
            )
        )

        return answer.strip()
