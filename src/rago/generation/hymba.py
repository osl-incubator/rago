"""Hymba model for text generation."""

from __future__ import annotations

import os
import subprocess
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typeguard import typechecked

from rago.generation.base import GenerationBase


@typechecked
class HymbaGen(GenerationBase):
    """HymbaGen for text generation using NVIDIA's Hymba-1.5B-Instruct model."""

    default_model_name = 'nvidia/Hymba-1.5B-Instruct'
    default_device_name = 'cuda'  # Hymba is optimized for CUDA

    def _validate(self) -> None:
        """Validate the model configuration."""
        if self.model_name != self.default_model_name:
            raise Exception(
                f'The given model {self.model_name} is not supported. '
                f'Only {self.default_model_name} is supported.'
            )

        if self.structured_output:
            warnings.warn(
                'Structured output is not supported yet in '
                f'{self.__class__.__name__}.'
            )

        if self.device_name == 'cpu':
            warnings.warn(
                'Hymba is optimized for CUDA. Using CPU may result in '
                'suboptimal performance.'
            )

    def _setup(self) -> None:
        """Set up the Hymba model and tokenizer."""
        # Check if CUDA is available
        if self.device_name == 'cuda' and not torch.cuda.is_available():
            raise Exception('CUDA is not available. Please check your installation.')

        # Download and run setup script if not already done
        setup_dir = Path.home() / '.rago' / 'hymba'
        setup_dir.mkdir(parents=True, exist_ok=True)
        setup_script = setup_dir / 'setup.sh'

        if not setup_script.exists():
            # Get Hugging Face token from environment
            hf_token = os.getenv('HUGGING_FACE_TOKEN')
            if not hf_token:
                raise Exception(
                    'HUGGING_FACE_TOKEN environment variable is required. '
                    'Please set it with your Hugging Face token.'
                )

            # Download setup script
            subprocess.run(
                [
                    'wget',
                    '--header=f"Authorization: Bearer {hf_token}"',
                    'https://huggingface.co/nvidia/Hymba-1.5B-Base/resolve/main/setup.sh',
                    '-O',
                    str(setup_script),
                ],
                check=True,
            )

            # Make script executable and run it
            setup_script.chmod(0o755)
            subprocess.run(['bash', str(setup_script)], check=True)

        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device_name == 'cuda' else torch.float32,
        ).to(self.device)

    def generate(self, query: str, context: list[str]) -> str:
        """Generate text using the Hymba model.

        Parameters
        ----------
        query : str
            The input query or prompt.
        context : list[str]
            Additional context information for the generation.

        Returns
        -------
        str
            Generated text based on query and context.
        """
        with torch.no_grad():
            # Format input text
            input_text = self.prompt_template.format(
                query=query, context=' '.join(context)
            )

            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
            ).to(self.device)

            # Generate response
            outputs = self.model.generate(
                **inputs,
                max_length=self.output_max_length,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.api_params,
            )

            # Decode and return response
            response = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

        if self.device_name == 'cuda':
            torch.cuda.empty_cache()

        return str(response) 