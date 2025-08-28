"""
Tests for the embedding service
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from backend.embedding_service import EmbeddingService, EmbeddingConfig


@pytest.fixture
def embedding_config():
    return EmbeddingConfig(
        openai_api_key="test-key",
        model_name="text-embedding-3-large",
        local_model_name="all-MiniLM-L6-v2",
        use_openai=False  # Use local for testing
    )


@pytest.fixture
def embedding_service(embedding_config):
    return EmbeddingService(embedding_config)


@pytest.mark.asyncio
async def test_generate_embedding_local(embedding_service):
    """Test local embedding generation"""
    text = "This is a test sentence for embedding generation."
    
    with patch.object(embedding_service.local_model, 'encode') as mock_encode:
        mock_encode.return_value = np.random.random((384,))  # Simulate embedding
        
        embedding = await embedding_service.generate_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        mock_encode.assert_called_once_with(text, convert_to_tensor=False)


@pytest.mark.asyncio
async def test_generate_embedding_openai(embedding_config):
    """Test OpenAI embedding generation"""
    embedding_config.use_openai = True
    service = EmbeddingService(embedding_config)
    
    text = "This is a test sentence for embedding generation."
    
    with patch('openai.OpenAI') as mock_openai:
        # Mock the OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=np.random.random((1536,)).tolist())]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        embedding = await service.generate_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 1536


@pytest.mark.asyncio
async def test_generate_embeddings_batch(embedding_service):
    """Test batch embedding generation"""
    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence."
    ]
    
    with patch.object(embedding_service.local_model, 'encode') as mock_encode:
        mock_encode.return_value = np.random.random((3, 384))  # Batch embeddings
        
        embeddings = await embedding_service.generate_embeddings_batch(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
        mock_encode.assert_called_once_with(texts, convert_to_tensor=False)


@pytest.mark.asyncio
async def test_chunk_text():
    """Test text chunking functionality"""
    from backend.embedding_service import chunk_text
    
    # Long text that should be chunked
    long_text = "This is a sentence. " * 100  # 500 words
    
    chunks = chunk_text(long_text, max_tokens=50, overlap_tokens=10)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 1  # Should be chunked
    assert all(isinstance(chunk, str) for chunk in chunks)
    
    # Check overlap
    if len(chunks) > 1:
        # There should be some overlap between consecutive chunks
        assert len(chunks[0]) > 0
        assert len(chunks[1]) > 0


@pytest.mark.asyncio
async def test_preprocess_text():
    """Test text preprocessing"""
    from backend.embedding_service import preprocess_text
    
    # Text with various issues
    dirty_text = "   This is a TEST with\n\n\nmultiple    spaces   and\ttabs.   "
    
    cleaned = preprocess_text(dirty_text)
    
    assert cleaned == "This is a TEST with multiple spaces and tabs."
    assert not cleaned.startswith(" ")
    assert not cleaned.endswith(" ")


def test_embedding_config_validation():
    """Test embedding configuration validation"""
    # Valid config
    config = EmbeddingConfig(
        openai_api_key="test-key",
        model_name="text-embedding-3-large"
    )
    assert config.openai_api_key == "test-key"
    assert config.model_name == "text-embedding-3-large"
    
    # Test default values
    assert config.batch_size == 100
    assert config.use_openai == True
    assert config.local_model_name == "all-MiniLM-L6-v2"


@pytest.mark.asyncio
async def test_embedding_service_initialization():
    """Test embedding service initialization"""
    config = EmbeddingConfig(
        openai_api_key="test-key",
        use_openai=False
    )
    
    service = EmbeddingService(config)
    
    assert service.config == config
    assert service.local_model is not None
    assert hasattr(service.local_model, 'encode')


@pytest.mark.asyncio
async def test_error_handling(embedding_service):
    """Test error handling in embedding generation"""
    
    with patch.object(embedding_service.local_model, 'encode') as mock_encode:
        mock_encode.side_effect = Exception("Model error")
        
        with pytest.raises(Exception):
            await embedding_service.generate_embedding("test text")


@pytest.mark.asyncio
async def test_empty_text_handling(embedding_service):
    """Test handling of empty text"""
    
    # Empty string should raise an error or return None
    with pytest.raises(ValueError):
        await embedding_service.generate_embedding("")
    
    # Whitespace only should also be handled
    with pytest.raises(ValueError):
        await embedding_service.generate_embedding("   ")


if __name__ == "__main__":
    pytest.main([__file__])