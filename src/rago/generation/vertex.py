"""Vertex AI Generation Model class for flexible Gemini-based text generation."""

from __future__ import annotations

from typing import Any, Optional, Type, cast

from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Part
import instructor

from pydantic import BaseModel
from typeguard import typechecked

from rago.generation.base import GenerationBase
from rago.extensions.cache import Cache


@typechecked
class VertexGen(GenerationBase):
    """Vertex AI generation model for text generation."""

    default_model_name = 'gemini-1.5-pro'
    default_api_params = {  # noqa: RUF012
        'top_p': 0.9,
        'top_k': 40,
    }

    project_id: Optional[str] = None
    location: str = 'us-central1'

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        prompt_template: str = '',
        output_max_length: int = 500,
        device: str = 'auto',
        structured_output: Optional[Type[BaseModel]] = None,
        system_message: str = '',
        api_params=None,
        api_key: str = '',
        project_id: str = '',
        location: str = 'us-central1',
        cache: Optional[Cache] = None,
        logs=None,
    ) -> None:
        """Initialize Vertex AI Generation class."""
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            prompt_template=prompt_template,
            output_max_length=output_max_length,
            device=device,
            structured_output=structured_output,
            system_message=system_message,
            api_params=api_params,
            api_key=api_key,
            cache=cache,
            logs=logs,
        )
        self.project_id = project_id
        self.location = location

    def _setup(self) -> None:
        """Set up the object with the initial parameters."""
        if not self.project_id:
            raise ValueError("project_id is required for Vertex AI")

        aiplatform.init(project=self.project_id, location=self.location)
        model = GenerativeModel(self.model_name)

        self.model = (
            instructor.from_vertexai(
                client=model,
                mode=instructor.Mode.VERTEXAI_TOOLS
                )
            if self.structured_output
            else model
        )

    def generate(
        self,
        query: str,
        context: list[str],
    ) -> str | BaseModel:
        """Generate text using Vertex AI's API with dynamic model support."""
        input_text = self.prompt_template.format(
            query=query, context=' '.join(context)
        )

        if not self.model:
            raise Exception('The model was not created.')

        api_params = (
            self.api_params if self.api_params else self.default_api_params
        )

        if self.structured_output:
            model_params = dict(
                contents=input_text,
                response_model=self.structured_output,
                max_output_tokens=self.output_max_length,
                temperature=self.temperature,
                **api_params,
            )

            if self.system_message:
                model_params["system_instruction"] = self.system_message

            response = self.model.completions.create(**model_params)

            self.logs['model_params'] = model_params
            return cast(BaseModel, response)
        else:
            model_params = dict(
                contents=input_text,
                generation_config={
                    "max_output_tokens": self.output_max_length,
                    "temperature": self.temperature,
                    **api_params,
                }
            )

            if self.system_message:
                model_params["system_instruction"] = self.system_message

            response = self.model.generate_content(**model_params)

            self.logs['model_params'] = model_params
            return cast(str, response.text.strip())

    def generate_with_image(
        self,
        query: str,
        image_uri: str,
        mime_type: str = "image/jpeg",
    ) -> str:
        """Generate text based on an image and query."""
        if not self.model:
            raise Exception('The model was not created.')

        api_params = (
            self.api_params if self.api_params else self.default_api_params
        )

        content = [
            Part.from_uri(image_uri, mime_type=mime_type),
            query,
        ]

        model_params = dict(
            contents=content,
            generation_config={
                "max_output_tokens": self.output_max_length,
                "temperature": self.temperature,
                **api_params,
            }
        )

        if self.system_message:
            model_params["system_instruction"] = self.system_message

        response = self.model.generate_content(**model_params)

        self.logs['model_params'] = model_params
        return response.text.strip()
