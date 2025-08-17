"""Llama generation module."""

from __future__ import annotations

import warnings

from copy import copy
from typing import Any

import instructor
import openai
import torch

from langdetect import detect
from ollama import Client as Ollama
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typeguard import typechecked

from rago.generation.base import GenerationBase
from rago.generation.openai import OpenAIGen


@typechecked
class LlamaGen(GenerationBase):
    """Llama Generation class."""

    default_model_name: str = 'meta-llama/Llama-3.2-1B'
    default_temperature: float = 0.5
    default_output_max_length: int = 500
    default_api_params = {
        'top_p': 1.0,
        'num_return_sequences': 1,
    }

    def _validate(self) -> None:
        """Raise an error if the initial parameters are not valid."""
        if not self.model_name.startswith('meta-llama/'):
            raise Exception(
                f'The given model name {self.model_name} is not provided '
                'by meta.'
            )

        if self.structured_output:
            warnings.warn(
                'Structured output is not supported yet in '
                f'{self.__class__.__name__}.'
            )

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=self.api_key
        )  # type: ignore

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.api_key,
            torch_dtype=torch.float16
            if self.device_name == 'cuda'
            else torch.float32,
        )

        self.generator = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device_name == 'cuda' else -1,
        )

    def generate(self, query: str, context: list[str]) -> str:
        """Generate text using Llama model with language support."""
        input_text = self.prompt_template.format(
            query=query, context=' '.join(context)
        )

        # Detect and set the language code for multilingual models (optional)
        language = str(detect(query)) or 'en'
        self.tokenizer.lang_code = language

        api_params = (
            self.api_params if self.api_params else self.default_api_params
        )

        # Generate the response with adjusted parameters

        model_params = dict(
            text_inputs=input_text,
            max_new_tokens=self.output_max_length,
            do_sample=True,
            temperature=self.temperature,
            eos_token_id=self.tokenizer.eos_token_id,
            **api_params,
        )
        response = self.generator(**model_params)

        self.logs['model_params'] = model_params

        # Extract and return the answer only
        answer = str(response[0].get('generated_text', ''))
        # Strip off any redundant text after the answer itself
        return answer.split('Answer:')[-1].strip()


@typechecked
class OllamaGen(GenerationBase):
    """Ollama Generation class for local inference via ollama-python."""

    default_model_name = 'llama3.2:1b'
    default_temperature: float = 0.5
    default_output_max_length: int = 500
    default_api_params: dict[str, Any] = {
        'base_url': 'http://localhost:11434/'
    }

    def _setup(self) -> None:
        """Instantiate the Ollama client."""
        self.api_params = copy(
            self.api_params if self.api_params else self.default_api_params
        )
        base_url = self.api_params.pop('base_url')

        self.model = Ollama(
            host=base_url, headers={'x-some-header': 'some-value'}
        )

    def generate(self, query: str, context: list[str]) -> str | BaseModel:
        """
        Generate text by sending a prompt to the local Ollama model.

        Parameters
        ----------
        query : str
            The user query.
        context : list[str]
            Augmented context strings.

        Returns
        -------
        str
            The generated response text.
        """
        input_text = self.prompt_template.format(
            query=query,
            context=' '.join(context),
        )

        messages = []
        if self.system_message:
            messages.append({'role': 'system', 'content': self.system_message})
        messages.append({'role': 'user', 'content': input_text})

        params = {
            'model': self.model_name,
            'messages': messages,
            **(self.api_params or {}),
        }
        response = self.model.chat(**params)
        return str(response.message.content).strip()


@typechecked
class OllamaOpenAIGen(OpenAIGen):
    """OllamaGen via the Ollama Python client."""

    default_model_name = 'llama3.2:1b'
    default_api_params: dict[str, Any] = {
        'base_url': 'http://localhost:11434/v1'
    }

    def _setup(self) -> None:
        self.api_params = copy(
            self.api_params if self.api_params else self.default_api_params
        )
        base_url = self.api_params.pop('base_url')

        model = openai.OpenAI(
            api_key='nokey',
            base_url=base_url,
        )

        self.model = (
            instructor.from_openai(model, mode=instructor.Mode.JSON)
            if self.structured_output
            else model
        )
