"""Tests for Rago package using declarative API with OpenAI GPT."""

import pytest


# Import the public pipeline interfaces (the user will only interact with these)
from rago import Augmented, Generation, Rago, Retrieval


@pytest.mark.skip_on_ci
def test_simple_rag_openai_gpt_general(
    animals_data: list[str], api_key_openai: str
) -> None:
    """
    Test the RAG pipeline using the declarative API with OpenAI's GPT.

    The pipeline is built as follows:
      - Retrieval: uses a string-based backend to return the source data.
      - Augmented: uses OpenAI augmentation with top_k=3.
      - Generation: uses OpenAI GPT (gpt-3.5-turbo) with temperature 0.

    The pipeline is executed with a query and the animal data as the source.
    """
    # Build the pipeline steps using the generic declarative classes.
    ret = Retrieval(backend='string')

    openai_params = dict(
        api_key=api_key_openai,
        backend='openai',
    )
    aug = Augmented(
        top_k=3, model_name='text-embedding-3-small', **openai_params
    )
    gen = Generation(
        temperature=0, model_name='gpt-3.5-turbo', **openai_params
    )

    # Compose the pipeline
    rag = Rago() + ret + aug + gen

    # Define the query and run the pipeline.
    query = 'Is there any animal larger than a dinosaur?'
    output = rag.run(query, animals_data)

    # Check if the generated response mentions "blue whale"
    assert 'blue whale' in output.result.lower(), (
        'Expected response to mention Blue Whale as a larger animal.'
    )
