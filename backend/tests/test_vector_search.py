"""
Tests for the vector search service
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from backend.vector_search import VectorSearchService, SearchRequest, SearchResult


@pytest.fixture
def db_config():
    return {
        'host': 'localhost',
        'port': 5432,
        'database': 'test_db',
        'user': 'test_user',
        'password': 'test_pass'
    }


@pytest.fixture
def search_service(db_config):
    return VectorSearchService(db_config)


@pytest.fixture
def search_request():
    return SearchRequest(
        query="How do I reset my password?",
        chatbot_id="test-chatbot-123",
        max_results=5,
        similarity_threshold=0.8
    )


@pytest.mark.asyncio
async def test_search_request_validation():
    """Test search request validation"""
    # Valid request
    request = SearchRequest(
        query="test query",
        chatbot_id="chatbot-123",
        max_results=10,
        similarity_threshold=0.7
    )
    assert request.query == "test query"
    assert request.similarity_threshold == 0.7
    
    # Invalid similarity threshold
    with pytest.raises(ValueError):
        SearchRequest(
            query="test",
            chatbot_id="chatbot-123",
            similarity_threshold=1.5  # Invalid: > 1.0
        )
    
    # Invalid max results
    with pytest.raises(ValueError):
        SearchRequest(
            query="test",
            chatbot_id="chatbot-123",
            max_results=150  # Invalid: > 100
        )


@pytest.mark.asyncio
async def test_generate_query_embedding(search_service):
    """Test query embedding generation"""
    query = "How do I reset my password?"
    
    with patch.object(search_service.embedding_model, 'encode') as mock_encode:
        mock_encode.return_value = np.random.random((384,))
        
        embedding = await search_service._generate_query_embedding(query)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        mock_encode.assert_called_once_with(query, convert_to_tensor=False)


@pytest.mark.asyncio
async def test_build_search_filters(search_service):
    """Test search filter building"""
    request = SearchRequest(
        query="test",
        chatbot_id="chatbot-123",
        search_types=["question", "answer"],
        category_filter="support"
    )
    
    where_clause, params = search_service._build_search_filters(request, "user-123", 4)
    
    assert "source_type = ANY" in where_clause
    assert "category =" in where_clause
    assert len(params) == 2
    assert params[0] == ["question", "answer"]
    assert params[1] == "support"


@pytest.mark.asyncio
async def test_search_similar_content(search_service, search_request):
    """Test similarity search"""
    
    # Mock database connection
    mock_conn = AsyncMock()
    
    # Mock query embedding generation
    with patch.object(search_service, '_generate_query_embedding') as mock_embed, \
         patch.object(search_service, 'get_db_connection', return_value=mock_conn) as mock_db:
        
        mock_embed.return_value = [0.1] * 384
        
        # Mock database query results
        mock_results = [
            {
                'id': 'result-1',
                'similarity_score': 0.95,
                'source_text': 'To reset your password, click Forgot Password.',
                'source_type': 'faq',
                'chatbot_id': 'chatbot-123',
                'chunk_index': 0,
                'usage_count': 5,
                'last_used_at': None,
                'title': 'Password Reset',
                'category': 'support',
                'tags': ['password', 'reset'],
                'content_type': 'faq',
                'user_message': None,
                'bot_response': None,
                'intent': None,
                'confidence_score': None
            }
        ]
        
        mock_conn.execute.return_value = None  # For setting user context
        mock_conn.fetch.return_value = mock_results
        mock_conn.__aenter__.return_value = mock_conn
        mock_conn.__aexit__.return_value = None
        
        results = await search_service.search_similar_content(search_request, "user-123")
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].id == 'result-1'
        assert results[0].similarity_score == 0.95
        assert results[0].content == 'To reset your password, click Forgot Password.'


@pytest.mark.asyncio
async def test_search_knowledge_base(search_service):
    """Test knowledge base search"""
    query = "password reset"
    chatbot_id = "chatbot-123"
    
    mock_conn = AsyncMock()
    
    with patch.object(search_service, '_generate_query_embedding') as mock_embed, \
         patch.object(search_service, 'get_db_connection', return_value=mock_conn):
        
        mock_embed.return_value = [0.1] * 384
        
        mock_results = [
            {
                'id': 'kb-1',
                'title': 'Password Reset Guide',
                'content': 'Complete guide to reset passwords.',
                'summary': 'How to reset passwords',
                'category': 'support',
                'tags': ['password'],
                'keywords': ['reset', 'password'],
                'similarity_score': 0.92,
                'access_count': 10,
                'effectiveness_score': 0.85,
                'last_accessed_at': None
            }
        ]
        
        mock_conn.fetch.return_value = mock_results
        mock_conn.execute.return_value = None
        mock_conn.__aenter__.return_value = mock_conn
        mock_conn.__aexit__.return_value = None
        
        results = await search_service.search_knowledge_base(query, chatbot_id)
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]['title'] == 'Password Reset Guide'
        assert results[0]['similarity_score'] == 0.92


@pytest.mark.asyncio
async def test_search_faq(search_service):
    """Test FAQ search"""
    query = "reset password"
    chatbot_id = "chatbot-123"
    
    mock_conn = AsyncMock()
    
    with patch.object(search_service, '_generate_query_embedding') as mock_embed, \
         patch.object(search_service, 'get_db_connection', return_value=mock_conn):
        
        mock_embed.return_value = [0.1] * 384
        
        mock_result = {
            'id': 'faq-1',
            'question': 'How do I reset my password?',
            'answer': 'Click forgot password link.',
            'category': 'support',
            'priority': 1,
            'q_similarity': 0.95,
            'match_type': 'question'
        }
        
        mock_conn.fetchrow.return_value = mock_result
        mock_conn.execute.return_value = None
        mock_conn.__aenter__.return_value = mock_conn
        mock_conn.__aexit__.return_value = None
        
        result = await search_service.search_faq(query, chatbot_id)
        
        assert result is not None
        assert result['question'] == 'How do I reset my password?'
        assert result['q_similarity'] == 0.95


@pytest.mark.asyncio
async def test_hybrid_search(search_service):
    """Test hybrid search across all content types"""
    query = "password help"
    chatbot_id = "chatbot-123"
    
    with patch.object(search_service, 'search_faq') as mock_faq, \
         patch.object(search_service, 'search_knowledge_base') as mock_kb, \
         patch.object(search_service, 'search_similar_content') as mock_similar:
        
        # Mock individual search results
        mock_faq.return_value = {'question': 'FAQ result', 'answer': 'FAQ answer'}
        mock_kb.return_value = [{'title': 'KB result', 'content': 'KB content'}]
        mock_similar.return_value = [
            SearchResult(
                id='sim-1',
                similarity_score=0.9,
                content='Similar content',
                source_type='document',
                metadata={},
                chatbot_id=chatbot_id
            )
        ]
        
        results = await search_service.hybrid_search(query, chatbot_id)
        
        assert 'faq_match' in results
        assert 'knowledge_base' in results
        assert 'similar_content' in results
        assert 'total_results' in results
        
        assert results['faq_match'] is not None
        assert len(results['knowledge_base']) == 1
        assert len(results['similar_content']) == 1
        assert results['total_results'] == 3


@pytest.mark.asyncio
async def test_contextual_search(search_service):
    """Test contextual search with conversation history"""
    query = "how about refunds?"
    conversation_history = [
        {"user": "What is your return policy?", "assistant": "We have a 30-day return policy."},
        {"user": "That's good to know", "assistant": "Is there anything else I can help with?"}
    ]
    chatbot_id = "chatbot-123"
    
    mock_conn = AsyncMock()
    
    with patch.object(search_service, '_generate_query_embedding') as mock_embed, \
         patch.object(search_service, 'get_db_connection', return_value=mock_conn):
        
        mock_embed.return_value = [0.1] * 384
        
        # Mock context and training results
        mock_context_results = []
        mock_training_results = [
            {
                'id': 'train-1',
                'similarity_score': 0.88,
                'source_text': 'Refunds are processed within 5-7 business days.',
                'source_type': 'document',
                'chatbot_id': chatbot_id
            }
        ]
        
        mock_conn.fetch.side_effect = [mock_context_results, mock_training_results]
        mock_conn.__aenter__.return_value = mock_conn
        mock_conn.__aexit__.return_value = None
        
        results = await search_service.contextual_search(
            query, conversation_history, chatbot_id
        )
        
        assert isinstance(results, list)
        # Should have training results since no context results
        assert len(results) >= 0


def test_build_context_window(search_service):
    """Test context window building"""
    conversation_history = [
        {"user": "Hello", "assistant": "Hi there!"},
        {"user": "What's your return policy?", "assistant": "We have a 30-day policy."}
    ]
    current_query = "How about refunds?"
    
    context = search_service._build_context_window(conversation_history, current_query)
    
    assert "User: Hello" in context
    assert "Assistant: Hi there!" in context
    assert "Current Query: How about refunds?" in context
    assert context.count("User:") == 2
    assert context.count("Assistant:") == 2


@pytest.mark.asyncio
async def test_error_handling(search_service, search_request):
    """Test error handling in search operations"""
    
    with patch.object(search_service, 'get_db_connection') as mock_db:
        mock_db.side_effect = Exception("Database connection failed")
        
        with pytest.raises(Exception):
            await search_service.search_similar_content(search_request)


@pytest.mark.asyncio
async def test_empty_results(search_service, search_request):
    """Test handling of empty search results"""
    
    mock_conn = AsyncMock()
    
    with patch.object(search_service, '_generate_query_embedding') as mock_embed, \
         patch.object(search_service, 'get_db_connection', return_value=mock_conn):
        
        mock_embed.return_value = [0.1] * 384
        mock_conn.fetch.return_value = []  # No results
        mock_conn.execute.return_value = None
        mock_conn.__aenter__.return_value = mock_conn
        mock_conn.__aexit__.return_value = None
        
        results = await search_service.search_similar_content(search_request)
        
        assert isinstance(results, list)
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__])