"""vLLM generation module for Rago."""

from __future__ import annotations

from typing import Any, cast, List, Optional, Type, Union

import warnings

from pydantic import BaseModel
from typeguard import typechecked
from vllm import LLMEngine, SamplingParams, VllmConfig

from rago.generation.base import GenerationBase


@typechecked
class VllmGen(GenerationBase):
    """
    vLLM Generation class.
    
    This class provides integration with vLLM, enabling high-performance
    inference with large language models.
    """

    default_model_name: str = 'meta-llama/Llama-3.2-1B'
    default_temperature: float = 0.5
    default_output_max_length: int = 500
    default_api_params = {  # noqa: RUF012
        'top_p': 0.9,
        'max_tokens': 500,
        'stop': None,
    }
    
    vllm_config: Optional[VllmConfig] = None
    llm_engine: Optional[LLMEngine] = None

    def _validate(self) -> None:
        """Raise an error if the initial parameters are not valid."""
        if self.structured_output:
            warnings.warn(
                'Structured output is currently not directly supported '
                f'in {self.__class__.__name__}. '
                'Output will be returned as raw text.'
            )

    def _setup(self) -> None:
        """Set up the vLLM engine with the initial parameters."""
        # Create a default config if not provided
        model_config = {
            "model": self.model_name,
            "trust_remote_code": True,
            "dtype": "float16" if self.device_name == "cuda" else "float32",
        }
        
        # Allow passing api_key as HF token if needed
        load_config = {}
        if self.api_key:
            load_config["download_dir"] = "./.cache/huggingface"
            # Token can be passed through the environment or constructor
            # This is mostly used for gated models
        
        # Handle device configuration
        device_config = {"device": self.device_name}
        
        # Create the vllm config
        self.vllm_config = VllmConfig(
            model_config=model_config,
            load_config=load_config,
            device_config=device_config,
        )
        
        # Initialize the LLM engine
        self.llm_engine = LLMEngine.from_vllm_config(self.vllm_config)
        
        # Store model and tokenizer for compatibility with other generators
        self.model = self.llm_engine
        self.tokenizer = self.llm_engine.get_tokenizer()

    def generate(
        self, 
        query: str, 
        context: list[str]
    ) -> Union[str, BaseModel]:
        """
        Generate text using vLLM with the provided query and context.
        
        Parameters
        ----------
        query : str
            The input query or prompt.
        context : list[str]
            Additional context information for the generation.
            
        Returns
        -------
        Union[str, BaseModel]
            Generated text based on query and context or structured output.
        """
        if not self.llm_engine:
            raise RuntimeError("LLM Engine not initialized. Call _setup() first.")
        
        # Format the input text using the prompt template
        input_text = self.prompt_template.format(
            query=query, context=' '.join(context)
        )

        # Extract generation parameters from api_params
        api_params = self.api_params or self.default_api_params
        
        # Convert api_params to SamplingParams
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.output_max_length,
            **api_params
        )
        
        # Log the model parameters
        self.logs['model_params'] = {
            'temperature': self.temperature,
            'max_tokens': self.output_max_length,
            **api_params
        }
        
        # Handle system message if provided
        if self.system_message:
            # Add system message to the prompt in a way compatible with vLLM
            input_text = f"{self.system_message}\n\n{input_text}"
        
        # Generate the text
        outputs = self.llm_engine.generate(input_text, sampling_params)
        
        # Extract the generated text
        result = outputs[0].outputs[0].text.strip()
        
        # Handle structured output if needed
        if self.structured_output:
            # Note: Direct structured output not supported by vLLM
            # This would require an additional parsing step or
            # using a different approach for structured output
            warnings.warn(
                "Structured output requested but vLLM doesn't support it directly. "
                "Returning raw text. Consider post-processing the output."
            )
            
        return result