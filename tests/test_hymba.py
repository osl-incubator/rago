"""Test Hymba model."""

import os
import pytest
from rago.generation import HymbaGen


@pytest.mark.skip_on_ci
def test_hymba_generation():
    """Test Hymba model generation."""
    # Skip if no Hugging Face token is available
    if not os.getenv('HUGGING_FACE_TOKEN'):
        pytest.skip('HUGGING_FACE_TOKEN environment variable is not set')

    # Initialize Hymba model
    model = HymbaGen(
        temperature=0.7,
        output_max_length=100,
    )

    # Test generation
    query = 'What is the capital of France?'
    context = ['Paris is the capital of France.']
    
    response = model.generate(query, context)
    
    assert isinstance(response, str)
    assert len(response) > 0
    assert 'Paris' in response.lower() 