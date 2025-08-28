"""
Pytest configuration and fixtures
"""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, Mock
import asyncpg


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_db_connection():
    """Mock database connection for testing"""
    mock_conn = AsyncMock()
    mock_conn.__aenter__.return_value = mock_conn
    mock_conn.__aexit__.return_value = None
    return mock_conn


@pytest.fixture
def test_db_config():
    """Test database configuration"""
    return {
        'host': 'localhost',
        'port': 5432,
        'database': 'test_chatbot_training',
        'user': 'test_user',
        'password': 'test_password'
    }


@pytest.fixture
def sample_training_data():
    """Sample training data for tests"""
    return {
        'chatbot_id': 'test-chatbot-123',
        'user_id': 'test-user-456',
        'content_type': 'faq',
        'raw_content': 'How do I reset my password? To reset your password, click the forgot password link.',
        'title': 'Password Reset FAQ',
        'category': 'support',
        'tags': ['password', 'reset', 'login'],
        'language': 'en'
    }


@pytest.fixture
def sample_conversation_pairs():
    """Sample conversation pairs for testing"""
    return [
        {
            'user_message': 'How do I reset my password?',
            'bot_response': 'To reset your password, go to the login page and click "Forgot Password".',
            'intent': 'password_reset',
            'confidence_score': 0.95,
            'category': 'support'
        },
        {
            'user_message': 'What are your business hours?',
            'bot_response': 'We are open Monday to Friday, 9 AM to 6 PM EST.',
            'intent': 'business_hours',
            'confidence_score': 0.92,
            'category': 'general'
        }
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing"""
    import numpy as np
    return {
        'openai_embedding': np.random.random((1536,)).tolist(),
        'local_embedding': np.random.random((384,)).tolist()
    }


@pytest.fixture
def temp_file():
    """Create a temporary file for testing file uploads"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("This is test content for file upload testing.\n")
        f.write("It contains multiple lines.\n")
        f.write("And some sample FAQ content:\n")
        f.write("Q: How do I contact support?\n")
        f.write("A: You can reach us at support@example.com\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing"""
    import csv
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'answer', 'category'])
        writer.writerow(['How do I reset my password?', 'Click forgot password link', 'support'])
        writer.writerow(['What are your hours?', 'Monday-Friday 9-6 EST', 'general'])
        writer.writerow(['How to cancel subscription?', 'Go to account settings', 'billing'])
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1] * 1536)]
    mock_client.embeddings.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing"""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.setex.return_value = True
    mock_redis.delete.return_value = 1
    mock_redis.keys.return_value = []
    return mock_redis


@pytest.fixture
def training_job_config():
    """Sample training job configuration"""
    return {
        'model_type': 'embedding_based',
        'base_model': 'all-MiniLM-L6-v2',
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 3,
        'validation_split': 0.2,
        'max_length': 512,
        'similarity_threshold': 0.8,
        'use_gpu': False,  # Disable GPU for testing
        'early_stopping': True,
        'save_best_model': True
    }


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer model"""
    mock_model = Mock()
    mock_model.encode.return_value = np.random.random((384,))
    return mock_model


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables"""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_NAME", "test_chatbot_training")
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("DB_PASSWORD", "test_password")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    from backend.vector_search import SearchResult
    
    return [
        SearchResult(
            id='result-1',
            similarity_score=0.95,
            content='To reset your password, go to the login page and click "Forgot Password".',
            source_type='faq',
            metadata={
                'title': 'Password Reset Instructions',
                'category': 'support',
                'tags': ['password', 'reset', 'login']
            },
            chatbot_id='test-chatbot-123'
        ),
        SearchResult(
            id='result-2',
            similarity_score=0.87,
            content='You can also reset your password from your account settings.',
            source_type='knowledge',
            metadata={
                'title': 'Account Settings Guide',
                'category': 'support',
                'tags': ['account', 'settings', 'password']
            },
            chatbot_id='test-chatbot-123'
        )
    ]


# Performance testing fixtures
@pytest.fixture
def large_text_dataset():
    """Generate a large dataset for performance testing"""
    texts = []
    for i in range(1000):
        text = f"This is test document number {i}. " * 10
        texts.append(text)
    return texts


@pytest.fixture
def benchmark_queries():
    """Common queries for benchmarking"""
    return [
        "How do I reset my password?",
        "What are your business hours?",
        "How to cancel my subscription?",
        "Where is my order?",
        "How to contact support?",
        "What is your refund policy?",
        "How to upgrade my plan?",
        "Technical issues with login",
        "Account security settings",
        "Payment method problems"
    ]


# Async test helpers
@pytest.fixture
async def async_test_client():
    """Create an async test client for API testing"""
    from fastapi.testclient import TestClient
    from backend.data_ingestion import app
    
    client = TestClient(app)
    yield client