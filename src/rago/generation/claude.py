"""Claude Generation Model class for text generation using Anthropic via Bedrock API."""

from __future__ import annotations

from typing import cast, Dict, Any

import boto3
import json
from pydantic import BaseModel
from typeguard import typechecked

from rago.generation.base import GenerationBase

@typechecked
class ClaudeGen(GenerationBase):
    """Claude generation model using Anthropic Claude via Amazon Bedrock API."""

    default_model_name: str = 'anthropic.claude-v2'
    default_temperature: float = 0.7
    default_output_max_length: int = 500
    default_api_params = {
        'top_p': 0.9,
        'max_tokens_to_sample': 500,
    }

    def _setup(self) -> None:
        """Ensure AWS credentials and Bedrock client are set up."""
        self.client = boto3.client('bedrock-runtime')

    def generate(self, query: str, context: list[str]) -> str | BaseModel:
        """Generate text using Anthropic Claude via Bedrock API."""
        input_text = self.prompt_template.format(
            query=query, context=' '.join(context)
        )

        api_params = self.api_params if self.api_params else self.default_api_params
        model_params: Dict[str, Any] = {
            'modelId': self.model_name,
            'prompt': input_text,
            'temperature': self.temperature,
            **api_params,
        }

        headers = {
            'Content-Type': 'application/json'
        }

        response = self.client.invoke_model(
            body=json.dumps(model_params),
            modelId=self.model_name,
        )
        result = json.loads(response['body'].read().decode('utf-8'))

        self.logs['generation'] = {
            'model': self.model_name,
            'input_text': input_text,
            'parameters': model_params,
        }

        generated_text = result.get('completion', '').strip()

        if self.structured_output:
            return cast(BaseModel, generated_text)

        return cast(str, generated_text)
