# src/rago/generation/llama3.py

"""Llama 3.2 1B classes for text generation."""

from __future__ import annotations

from typing import cast

import torch

from langdetect import detect
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class LlamaV32M1BGen(GenerationBase):
    """Llama 3.2 1B Generation class."""

    def __init__(
        self,
        model_name: str = 'meta-llama/Llama-3.2-1B',
        output_max_length: int = 500,
        apikey: str = '',
        device: str = 'auto',
    ) -> None:
        """Initialize LlamaV32M1BGen.

        Parameters
        ----------
        model_name : str
            The name of the model to use.
        output_max_length : int
            Maximum length of the generated output.
        apikey : str
            API key for accessing the pretrained models.
        device : str
            Device for running the model ('cpu', 'cuda', or 'auto').
        """
        super().__init__(
            model_name=model_name, output_max_length=output_max_length
        )

        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=apikey
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=apikey,
            torch_dtype=torch.float16
            if self.device.type == 'cuda'
            else torch.float32,
        )

        self.generator = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device.type == 'cuda' else -1,
        )

    def generate(
        self,
        query: str,
        context: list[str],
    ) -> str:
        """
        Generate text using Llama 3.2 1B model with language support.

        Parameters
        ----------
        query : str
            The input query or prompt.
        context : list[str]
            Contextual information for the query.

        Return
        ------

        str
            The generated response.
        """
        input_text = f"Question: {query} Context: {' '.join(context)}"
        language = str(detect(query)) or 'en'

        self.tokenizer.lang_code = language

        response = self.generator(
            input_text,
            max_length=self.output_max_length,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=1,
        )

        return cast(str, response[0].get('generated_text', ''))
