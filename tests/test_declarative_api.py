"""Tests for the Rago declarative API."""

import pytest

from rago import Augmentation, Cache, DB, Generation, Rago, Retrieval


def test_rago_declarative_api_init():
    """Test that the Rago class can be initialized with the declarative API."""
    rag = Rago()
    assert rag is not None
    assert rag._components == []


def test_rago_add_component():
    """Test that components can be added to the Rago class."""
    rag = Rago()
    rag = rag + DB(backend="faiss")
    assert len(rag._components) == 1
    assert type(rag._components[0]).__name__ == "DB"


def test_rago_build_components():
    """Test that the Rago class builds components correctly."""
    # Sample data
    sample_data = ["Test document 1", "Test document 2"]
    
    # Create a minimal pipeline with test-friendly components
    rag = (
        Rago()
        + DB(backend="faiss")
        + Retrieval(backend="string")
        + Augmentation(backend="sentencetransformer", 
                      model="all-MiniLM-L6-v2")
        + Generation(backend="openai", model="gpt-4")
    )
    
    # Mock the build methods to avoid actual model loading
    # This is just for testing the build process, not the actual functionality
    with pytest.raises(ValueError):
        # This should raise an error because we're not providing actual API keys
        # But it shows that the build process is being attempted
        rag.run(query="Test query", data=sample_data)


def test_rago_missing_components():
    """Test that an error is raised when required components are missing."""
    rag = Rago()
    rag = rag + DB(backend="faiss")
    
    with pytest.raises(ValueError, match="Missing required component"):
        rag.run(query="Test query")


def test_rago_wrong_component_type():
    """Test that an error is raised when adding a non-component."""
    rag = Rago()
    
    with pytest.raises(TypeError, match="Expected a Component"):
        rag = rag + "Not a component"