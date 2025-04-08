"""Rago is Retrieval Augmented Generation lightweight framework."""

from __future__ import annotations

from typing import Any, Optional, Type, Union, cast

from pydantic import BaseModel
from typeguard import typechecked

from rago.augmented.base import AugmentedBase
from rago.augmented.db import DBBase, FaissDB
from rago.base import RagoBase
from rago.extensions.cache import Cache
from rago.generation.base import GenerationBase
from rago.retrieval.base import RetrievalBase


@typechecked
class Rago:
    """RAG class."""

    retrieval: Optional[RetrievalBase] = None
    augmented: Optional[AugmentedBase] = None
    generation: Optional[GenerationBase] = None
    db: Optional[DBBase] = None
    cache: Optional[Cache] = None
    logs: dict[str, dict[str, Any]]
    _components: list[Any] = []  # Store components for declarative API

    def __init__(
        self,
        retrieval: Optional[RetrievalBase] = None,
        augmented: Optional[AugmentedBase] = None,
        generation: Optional[GenerationBase] = None,
    ) -> None:
        """Initialize the RAG structure.

        This constructor supports both the original procedural API and
        the new declarative API.

        Parameters
        ----------
        retrieval : Optional[RetrievalBase]
            The retrieval component used to fetch relevant data based
            on the query.
        augmented : Optional[AugmentedBase]
            The augmentation module responsible for enriching the
            retrieved data.
        generation : Optional[GenerationBase]
            The text generation model used to generate a response based
            on the query and augmented data.
        """
        self.retrieval = retrieval
        self.augmented = augmented
        self.generation = generation
        self._components = []
        
        # Initialize logs dictionary for all components
        self.logs = {}
        if retrieval:
            self.logs['retrieval'] = retrieval.logs
        if augmented:
            self.logs['augmented'] = augmented.logs
        if generation:
            self.logs['generation'] = generation.logs

    def __add__(self, component: Any) -> Rago:
        """Add a component to the RAG pipeline.
        
        This allows for the declarative API style by using the + operator
        to add components to the pipeline.
        
        Parameters
        ----------
        component : Any
            The component to add to the pipeline.
            
        Returns
        -------
        Rago
            The updated Rago instance for method chaining.
        """
        from rago.components import Component
        
        if not isinstance(component, Component):
            raise TypeError(f"Expected a Component, got {type(component).__name__}")
        
        self._components.append(component)
        return self

    def prompt(self, query: str, device: str = 'auto') -> Union[str, BaseModel]:
        """Run the pipeline for a specific prompt using the procedural API.

        Parameters
        ----------
        query : str
            The query or prompt from the user.
        device : str (default 'auto')
            Device for generation (e.g., 'auto', 'cpu', 'cuda'), by
            default 'auto'.

        Returns
        -------
        Union[str, BaseModel]
            Generated text based on the query and augmented data.
        """
        if not all([self.retrieval, self.augmented, self.generation]):
            raise ValueError(
                "Cannot run prompt with procedural API when components are missing. "
                "Either provide all components through the constructor or use the "
                "declarative API and call run() instead."
            )
            
        ret_data = self.retrieval.get(query)
        self.logs['retrieval']['result'] = ret_data

        aug_data = self.augmented.search(query, ret_data)
        self.logs['augmented']['result'] = aug_data

        gen_data = self.generation.generate(query, context=aug_data)
        self.logs['generation']['result'] = gen_data

        return gen_data

    def run(self, query: str, data: Any = None, device: str = 'auto') -> Union[str, BaseModel]:
        """Run the pipeline for a specific prompt using the declarative API.
        
        This method builds the pipeline from the declarative components
        and executes it.
        
        Parameters
        ----------
        query : str
            The query or prompt from the user.
        data : Any, optional
            The data source to use for retrieval.
        device : str (default 'auto')
            Device for generation (e.g., 'auto', 'cpu', 'cuda'), by
            default 'auto'.
            
        Returns
        -------
        Union[str, BaseModel]
            Generated text based on the query and augmented data.
        """
        # If components exist, we need to build them
        if self._components:
            # Check if all required components are present
            component_types = [type(comp).__name__ for comp in self._components]
            required_components = ['Retrieval', 'Augmentation', 'Generation']
            
            for required in required_components:
                if required not in component_types:
                    raise ValueError(f"Missing required component: {required}")
            
            # Build components in the right order
            for component in self._components:
                comp_type = type(component).__name__
                
                if comp_type == 'DB':
                    self.db = component.build()
                elif comp_type == 'Cache':
                    self.cache = component.build()
                elif comp_type == 'Retrieval':
                    self.retrieval = component.build(source=data)
                    if self.cache:
                        self.retrieval.cache = self.cache
                    self.logs['retrieval'] = self.retrieval.logs
                elif comp_type == 'Augmentation':
                    self.augmented = component.build(db=self.db)
                    if self.cache:
                        self.augmented.cache = self.cache
                    self.logs['augmented'] = self.augmented.logs
                elif comp_type == 'Generation':
                    self.generation = component.build()
                    if self.cache:
                        self.generation.cache = self.cache
                    self.logs['generation'] = self.generation.logs
        
        # Make sure all required components are available
        if not all([self.retrieval, self.augmented, self.generation]):
            raise ValueError(
                "Missing required components. Ensure that Retrieval, Augmentation, "
                "and Generation components are added to the pipeline."
            )
            
        # Now run the pipeline
        ret_data = self.retrieval.get(query)
        self.logs['retrieval']['result'] = ret_data

        aug_data = self.augmented.search(query, ret_data)
        self.logs['augmented']['result'] = aug_data

        gen_data = self.generation.generate(query, context=aug_data)
        self.logs['generation']['result'] = gen_data

        return gen_data