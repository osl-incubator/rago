"""Component classes for the Rago declarative API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Type, TypeVar, cast

from pydantic import BaseModel

from rago.augmented.base import AugmentedBase
from rago.augmented.db import DBBase, FaissDB
from rago.base import RagoBase
from rago.extensions.cache import Cache
from rago.generation.base import GenerationBase
from rago.retrieval.base import RetrievalBase

T = TypeVar('T', bound='Component')


class Component(ABC):
    """Base component class for the declarative API."""

    @abstractmethod
    def build(self) -> Any:
        """Build the actual implementation object."""
        ...


class DB(Component):
    """Database component for the declarative API."""

    def __init__(
        self, 
        backend: str = "faiss", 
        top_k: int = 5,
        **kwargs: Any
    ) -> None:
        """Initialize DB component.
        
        Parameters
        ----------
        backend : str
            The database backend to use.
        top_k : int
            Number of top results to return.
        **kwargs : Any
            Additional parameters to pass to the DB implementation.
        """
        self.backend = backend
        self.top_k = top_k
        self.kwargs = kwargs

    def build(self) -> DBBase:
        """Build the DB implementation."""
        if self.backend.lower() == "faiss":
            return FaissDB(top_k=self.top_k, **self.kwargs)
        # Add more backends as needed
        raise ValueError(f"Unsupported DB backend: {self.backend}")


class Cache(Component):
    """Cache component for the declarative API."""

    def __init__(
        self, 
        backend: str = "file", 
        path: str = ".cache",
        **kwargs: Any
    ) -> None:
        """Initialize Cache component.
        
        Parameters
        ----------
        backend : str
            The cache backend to use.
        path : str
            Path to store cache files.
        **kwargs : Any
            Additional parameters to pass to the Cache implementation.
        """
        self.backend = backend
        self.path = path
        self.kwargs = kwargs

    def build(self) -> Cache:
        """Build the Cache implementation."""
        if self.backend.lower() == "file":
            return Cache(path=self.path, **self.kwargs)
        # Add more backends as needed
        raise ValueError(f"Unsupported Cache backend: {self.backend}")


class Retrieval(Component):
    """Retrieval component for the declarative API."""

    def __init__(
        self, 
        backend: str,
        **kwargs: Any
    ) -> None:
        """Initialize Retrieval component.
        
        Parameters
        ----------
        backend : str
            The retrieval backend to use.
        **kwargs : Any
            Additional parameters to pass to the Retrieval implementation.
        """
        self.backend = backend
        self.kwargs = kwargs
        
    def build(self, source: Any = None) -> RetrievalBase:
        """Build the Retrieval implementation.
        
        Parameters
        ----------
        source : Any
            The data source to retrieve from.
            
        Returns
        -------
        RetrievalBase
            The instantiated retrieval object.
        """
        # This will need to be expanded with actual implementations
        from rago.retrieval import StringRet
        
        if self.backend.lower() == "dummy" or self.backend.lower() == "string":
            return StringRet(source=source or [], **self.kwargs)
        
        # Import and return appropriate implementation based on backend
        raise ValueError(f"Unsupported Retrieval backend: {self.backend}")


class Augmentation(Component):
    """Augmentation component for the declarative API."""

    def __init__(
        self, 
        backend: str,
        model: str = "",
        top_k: int = 5,
        **kwargs: Any
    ) -> None:
        """Initialize Augmentation component.
        
        Parameters
        ----------
        backend : str
            The augmentation backend to use.
        model : str
            The model to use for embeddings.
        top_k : int
            Number of top results to return.
        **kwargs : Any
            Additional parameters to pass to the Augmentation implementation.
        """
        self.backend = backend
        self.model = model
        self.top_k = top_k
        self.kwargs = kwargs
        
    def build(self, db: Optional[DBBase] = None) -> AugmentedBase:
        """Build the Augmentation implementation.
        
        Parameters
        ----------
        db : Optional[DBBase]
            The database to use for augmentation.
            
        Returns
        -------
        AugmentedBase
            The instantiated augmentation object.
        """
        # Import required implementations based on backend
        if self.backend.lower() == "openai":
            from rago.augmented import OpenAIAug
            return OpenAIAug(
                model_name=self.model, 
                db=db or FaissDB(),
                top_k=self.top_k,
                **self.kwargs
            )
        elif self.backend.lower() == "sentencetransformer":
            from rago.augmented import SentenceTransformerAug
            return SentenceTransformerAug(
                model_name=self.model, 
                db=db or FaissDB(),
                top_k=self.top_k,
                **self.kwargs
            )
        
        # Add more backends as needed
        raise ValueError(f"Unsupported Augmentation backend: {self.backend}")


class Generation(Component):
    """Generation component for the declarative API."""

    def __init__(
        self, 
        backend: str,
        model: str = "",
        temperature: float = 0.5,
        prompt_template: str = "",
        output_max_length: int = 500,
        system_message: str = "",
        structured_output: Optional[Type[BaseModel]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize Generation component.
        
        Parameters
        ----------
        backend : str
            The generation backend to use.
        model : str
            The model to use for generation.
        temperature : float
            The temperature parameter for generation.
        prompt_template : str
            The template for generating prompts.
        output_max_length : int
            Maximum length of the generated output.
        system_message : str
            System message to include in the prompt.
        structured_output : Optional[Type[BaseModel]]
            Type for structured output.
        **kwargs : Any
            Additional parameters to pass to the Generation implementation.
        """
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.prompt_template = prompt_template
        self.output_max_length = output_max_length
        self.system_message = system_message
        self.structured_output = structured_output
        self.kwargs = kwargs
        
    def build(self) -> GenerationBase:
        """Build the Generation implementation."""
        # Import required implementations based on backend
        if self.backend.lower() == "openai":
            from rago.generation import OpenAIGen
            if not self.kwargs.get('api_key') and 'OPENAI_API_KEY' not in os.environ:
                raise ValueError(
                    "OpenAI API key is required. Either pass it as 'api_key' parameter "
                    "or set the OPENAI_API_KEY environment variable."
                )
            return OpenAIGen(
                model_name=self.model,
                temperature=self.temperature,
                prompt_template=self.prompt_template,
                output_max_length=self.output_max_length,
                system_message=self.system_message,
                structured_output=self.structured_output,
                **self.kwargs
            )
        elif self.backend.lower() == "llama":
            from rago.generation import LlamaGen
            return LlamaGen(
                model_name=self.model,
                temperature=self.temperature,
                prompt_template=self.prompt_template,
                output_max_length=self.output_max_length,
                system_message=self.system_message,
                structured_output=self.structured_output,
                **self.kwargs
            )
        elif self.backend.lower() == "gemini":
            from rago.generation import GeminiGen
            return GeminiGen(
                model_name=self.model,
                temperature=self.temperature,
                prompt_template=self.prompt_template,
                output_max_length=self.output_max_length,
                system_message=self.system_message,
                structured_output=self.structured_output,
                **self.kwargs
            )
        
        # Add more backends as needed
        raise ValueError(f"Unsupported Generation backend: {self.backend}")