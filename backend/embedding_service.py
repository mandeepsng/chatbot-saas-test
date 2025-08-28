#!/usr/bin/env python3
"""
ChatFlow AI - Vector Embedding Generation Service
Handles conversion of text data to vector embeddings for training
"""

import asyncio
import json
import logging
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import uuid

import openai
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential
import redis
from celery import Celery
from pydantic import BaseModel, ValidationError

# Configuration
OPENAI_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 1536
MAX_TOKENS_PER_REQUEST = 8192
BATCH_SIZE = 100
REDIS_URL = "redis://localhost:6379/0"
CELERY_BROKER_URL = "redis://localhost:6379/1"

# Initialize services
openai.api_key = "your-openai-api-key"  # Load from environment
redis_client = redis.Redis.from_url(REDIS_URL)
celery_app = Celery('embedding_service', broker=CELERY_BROKER_URL)

# Local embedding model fallback
local_model = None

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingRequest:
    """Request for generating embeddings"""
    texts: List[str]
    chatbot_id: str
    source_type: str
    training_data_id: Optional[str] = None
    conversation_pair_id: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass 
class EmbeddingResult:
    """Result of embedding generation"""
    embeddings: List[List[float]]
    tokens_used: int
    model_used: str
    processing_time: float
    success: bool
    error_message: Optional[str] = None

class EmbeddingService:
    """Main service for generating and managing vector embeddings"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self._load_local_model()
        
    def _load_local_model(self):
        """Load local sentence transformer model as fallback"""
        global local_model
        try:
            local_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Local embedding model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load local model: {e}")
    
    async def get_db_connection(self):
        """Get database connection with async context manager"""
        return psycopg2.connect(**self.db_config)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))
    
    def chunk_text(self, text: str, max_tokens: int = 512, overlap: int = 50) -> List[Dict]:
        """Split text into chunks with overlap for embedding"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        chunk_index = 0
        
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                'text': chunk_text,
                'chunk_index': chunk_index,
                'chunk_size': len(chunk_tokens),
                'overlap_tokens': overlap if chunk_index > 0 else 0,
                'start_token': start,
                'end_token': end
            })
            
            chunk_index += 1
            start = end - overlap if end < len(tokens) else end
            
        return chunks
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_openai_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """Generate embeddings using OpenAI API with retry logic"""
        try:
            start_time = time.time()
            
            # Filter out empty texts
            valid_texts = [text.strip() for text in texts if text.strip()]
            if not valid_texts:
                return [], 0
            
            response = await openai.Embedding.acreate(
                model=OPENAI_MODEL,
                input=valid_texts,
                encoding_format="float"
            )
            
            embeddings = [item['embedding'] for item in response['data']]
            tokens_used = response['usage']['total_tokens']
            
            processing_time = time.time() - start_time
            logger.info(f"Generated {len(embeddings)} OpenAI embeddings in {processing_time:.2f}s")
            
            return embeddings, tokens_used
            
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise
    
    def generate_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model as fallback"""
        if not local_model:
            raise ValueError("Local embedding model not available")
        
        start_time = time.time()
        valid_texts = [text.strip() for text in texts if text.strip()]
        
        if not valid_texts:
            return []
        
        embeddings = local_model.encode(valid_texts, convert_to_numpy=True)
        
        # Pad/truncate to match OpenAI embedding dimension
        if embeddings.shape[1] != EMBEDDING_DIMENSION:
            if embeddings.shape[1] < EMBEDDING_DIMENSION:
                # Pad with zeros
                padding = np.zeros((embeddings.shape[0], EMBEDDING_DIMENSION - embeddings.shape[1]))
                embeddings = np.concatenate([embeddings, padding], axis=1)
            else:
                # Truncate
                embeddings = embeddings[:, :EMBEDDING_DIMENSION]
        
        processing_time = time.time() - start_time
        logger.info(f"Generated {len(embeddings)} local embeddings in {processing_time:.2f}s")
        
        return embeddings.tolist()
    
    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Generate embeddings with fallback strategy"""
        start_time = time.time()
        
        try:
            # Try OpenAI first
            embeddings, tokens_used = await self.generate_openai_embeddings(request.texts)
            model_used = OPENAI_MODEL
            
        except Exception as e:
            logger.warning(f"OpenAI embedding failed, falling back to local model: {e}")
            
            try:
                embeddings = self.generate_local_embeddings(request.texts)
                tokens_used = sum(self.count_tokens(text) for text in request.texts)
                model_used = "sentence-transformers/all-MiniLM-L6-v2"
                
            except Exception as fallback_error:
                return EmbeddingResult(
                    embeddings=[],
                    tokens_used=0,
                    model_used="none",
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=f"Both OpenAI and local embedding failed: {fallback_error}"
                )
        
        return EmbeddingResult(
            embeddings=embeddings,
            tokens_used=tokens_used,
            model_used=model_used,
            processing_time=time.time() - start_time,
            success=True
        )
    
    async def store_embeddings(self, request: EmbeddingRequest, result: EmbeddingResult) -> List[str]:
        """Store generated embeddings in database"""
        if not result.success:
            raise ValueError(f"Cannot store failed embedding result: {result.error_message}")
        
        conn = await self.get_db_connection()
        try:
            with conn.cursor() as cursor:
                embedding_ids = []
                
                for i, (text, embedding) in enumerate(zip(request.texts, result.embeddings)):
                    embedding_id = str(uuid.uuid4())
                    
                    cursor.execute("""
                        INSERT INTO training_embeddings (
                            id, chatbot_id, training_data_id, conversation_pair_id,
                            embedding, embedding_model, source_text, source_type,
                            chunk_index, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        embedding_id,
                        request.chatbot_id,
                        request.training_data_id,
                        request.conversation_pair_id,
                        embedding,
                        result.model_used,
                        text,
                        request.source_type,
                        i,
                        datetime.utcnow()
                    ))
                    
                    embedding_ids.append(embedding_id)
                
                conn.commit()
                logger.info(f"Stored {len(embedding_ids)} embeddings for chatbot {request.chatbot_id}")
                return embedding_ids
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store embeddings: {e}")
            raise
        finally:
            conn.close()
    
    async def process_training_data(self, training_data_id: str) -> Dict[str, Any]:
        """Process training data and generate embeddings"""
        conn = await self.get_db_connection()
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get training data
                cursor.execute("""
                    SELECT td.*, c.id as chatbot_id 
                    FROM training_data td
                    JOIN chatbots c ON td.chatbot_id = c.id
                    WHERE td.id = %s AND td.status = 'pending'
                """, (training_data_id,))
                
                training_data = cursor.fetchone()
                if not training_data:
                    raise ValueError(f"Training data {training_data_id} not found or not pending")
                
                # Update status to processing
                cursor.execute("""
                    UPDATE training_data 
                    SET status = 'processing', updated_at = %s
                    WHERE id = %s
                """, (datetime.utcnow(), training_data_id))
                conn.commit()
                
                # Process content based on type
                if training_data['content_type'] == 'conversation':
                    result = await self._process_conversation_data(training_data)
                elif training_data['content_type'] == 'document':
                    result = await self._process_document_data(training_data)
                elif training_data['content_type'] == 'faq':
                    result = await self._process_faq_data(training_data)
                else:
                    result = await self._process_generic_data(training_data)
                
                # Update training data status
                cursor.execute("""
                    UPDATE training_data 
                    SET status = %s, processed_content = %s, updated_at = %s
                    WHERE id = %s
                """, (
                    'completed' if result['success'] else 'failed',
                    json.dumps(result.get('processed_content', {})),
                    datetime.utcnow(),
                    training_data_id
                ))
                conn.commit()
                
                return result
                
        except Exception as e:
            # Mark as failed
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE training_data 
                    SET status = 'failed', processing_error = %s, updated_at = %s
                    WHERE id = %s
                """, (str(e), datetime.utcnow(), training_data_id))
                conn.commit()
            
            logger.error(f"Failed to process training data {training_data_id}: {e}")
            raise
        finally:
            conn.close()
    
    async def _process_conversation_data(self, training_data: Dict) -> Dict[str, Any]:
        """Process conversation data into Q&A pairs"""
        raw_content = training_data['raw_content']
        
        # Parse conversation format (assuming JSON format)
        try:
            conversations = json.loads(raw_content)
        except json.JSONDecodeError:
            # Try to parse as plain text conversations
            conversations = self._parse_text_conversations(raw_content)
        
        conn = await self.get_db_connection()
        pairs_created = 0
        embeddings_created = 0
        
        try:
            for conversation in conversations:
                if 'messages' not in conversation:
                    continue
                
                messages = conversation['messages']
                context = ""
                
                for i in range(len(messages) - 1):
                    if messages[i]['role'] == 'user' and messages[i + 1]['role'] == 'assistant':
                        user_message = messages[i]['content']
                        bot_response = messages[i + 1]['content']
                        
                        # Create conversation pair
                        with conn.cursor() as cursor:
                            pair_id = str(uuid.uuid4())
                            cursor.execute("""
                                INSERT INTO conversation_pairs (
                                    id, chatbot_id, training_data_id, user_message, 
                                    bot_response, context, created_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (
                                pair_id, training_data['chatbot_id'], training_data['id'],
                                user_message, bot_response, context, datetime.utcnow()
                            ))
                            pairs_created += 1
                        
                        # Generate embeddings for question and answer
                        request = EmbeddingRequest(
                            texts=[user_message, bot_response],
                            chatbot_id=training_data['chatbot_id'],
                            source_type='conversation',
                            conversation_pair_id=pair_id
                        )
                        
                        result = await self.generate_embeddings(request)
                        if result.success:
                            await self.store_embeddings(request, result)
                            embeddings_created += len(result.embeddings)
                        
                        # Update context for next iteration
                        context += f"User: {user_message}\nBot: {bot_response}\n"
            
            conn.commit()
            
            return {
                'success': True,
                'pairs_created': pairs_created,
                'embeddings_created': embeddings_created,
                'processed_content': {
                    'conversation_pairs': pairs_created,
                    'total_embeddings': embeddings_created
                }
            }
            
        except Exception as e:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    async def _process_document_data(self, training_data: Dict) -> Dict[str, Any]:
        """Process document data into chunks with embeddings"""
        content = training_data['raw_content']
        
        # Chunk the document
        chunks = self.chunk_text(content, max_tokens=512, overlap=50)
        
        # Generate embeddings for chunks
        texts = [chunk['text'] for chunk in chunks]
        request = EmbeddingRequest(
            texts=texts,
            chatbot_id=training_data['chatbot_id'],
            source_type='document_chunk',
            training_data_id=training_data['id']
        )
        
        result = await self.generate_embeddings(request)
        
        if result.success:
            await self.store_embeddings(request, result)
            
            return {
                'success': True,
                'chunks_created': len(chunks),
                'embeddings_created': len(result.embeddings),
                'processed_content': {
                    'chunks': len(chunks),
                    'total_tokens': sum(chunk['chunk_size'] for chunk in chunks),
                    'model_used': result.model_used
                }
            }
        else:
            return {
                'success': False,
                'error': result.error_message
            }
    
    async def _process_faq_data(self, training_data: Dict) -> Dict[str, Any]:
        """Process FAQ data into structured Q&A entries"""
        raw_content = training_data['raw_content']
        
        try:
            faq_data = json.loads(raw_content)
        except json.JSONDecodeError:
            # Parse as plain text FAQ
            faq_data = self._parse_text_faq(raw_content)
        
        conn = await self.get_db_connection()
        faqs_created = 0
        embeddings_created = 0
        
        try:
            for faq_item in faq_data:
                question = faq_item.get('question', '')
                answer = faq_item.get('answer', '')
                category = faq_item.get('category', 'general')
                
                if not question or not answer:
                    continue
                
                # Generate embeddings for question and answer
                request = EmbeddingRequest(
                    texts=[question, answer],
                    chatbot_id=training_data['chatbot_id'],
                    source_type='faq'
                )
                
                result = await self.generate_embeddings(request)
                
                if result.success:
                    # Store FAQ entry
                    with conn.cursor() as cursor:
                        faq_id = str(uuid.uuid4())
                        cursor.execute("""
                            INSERT INTO faq_entries (
                                id, chatbot_id, question, answer, category,
                                question_embedding, answer_embedding, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            faq_id, training_data['chatbot_id'], question, answer, category,
                            result.embeddings[0], result.embeddings[1], datetime.utcnow()
                        ))
                        faqs_created += 1
                        embeddings_created += 2
            
            conn.commit()
            
            return {
                'success': True,
                'faqs_created': faqs_created,
                'embeddings_created': embeddings_created,
                'processed_content': {
                    'faq_entries': faqs_created,
                    'total_embeddings': embeddings_created
                }
            }
            
        except Exception as e:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    async def _process_generic_data(self, training_data: Dict) -> Dict[str, Any]:
        """Process generic text data"""
        content = training_data['raw_content']
        
        # Create chunks for generic content
        chunks = self.chunk_text(content, max_tokens=512, overlap=50)
        texts = [chunk['text'] for chunk in chunks]
        
        request = EmbeddingRequest(
            texts=texts,
            chatbot_id=training_data['chatbot_id'],
            source_type='knowledge',
            training_data_id=training_data['id']
        )
        
        result = await self.generate_embeddings(request)
        
        if result.success:
            await self.store_embeddings(request, result)
            
            return {
                'success': True,
                'embeddings_created': len(result.embeddings),
                'processed_content': {
                    'chunks': len(chunks),
                    'model_used': result.model_used
                }
            }
        else:
            return {
                'success': False,
                'error': result.error_message
            }
    
    def _parse_text_conversations(self, text: str) -> List[Dict]:
        """Parse plain text conversations into structured format"""
        # Simple parser for text format like:
        # User: question
        # Bot: response
        lines = text.strip().split('\n')
        conversations = []
        current_conversation = {'messages': []}
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_conversation['messages']:
                    conversations.append(current_conversation)
                    current_conversation = {'messages': []}
                continue
            
            if line.startswith('User:') or line.startswith('user:'):
                content = line[5:].strip()
                current_conversation['messages'].append({
                    'role': 'user',
                    'content': content
                })
            elif line.startswith('Bot:') or line.startswith('bot:') or line.startswith('Assistant:'):
                content = line.split(':', 1)[1].strip()
                current_conversation['messages'].append({
                    'role': 'assistant',
                    'content': content
                })
        
        if current_conversation['messages']:
            conversations.append(current_conversation)
        
        return conversations
    
    def _parse_text_faq(self, text: str) -> List[Dict]:
        """Parse plain text FAQ into structured format"""
        # Simple parser for FAQ format like:
        # Q: question
        # A: answer
        lines = text.strip().split('\n')
        faqs = []
        current_faq = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                if 'question' in current_faq and 'answer' in current_faq:
                    faqs.append(current_faq)
                    current_faq = {}
                continue
            
            if line.startswith('Q:') or line.startswith('Question:'):
                current_faq['question'] = line.split(':', 1)[1].strip()
            elif line.startswith('A:') or line.startswith('Answer:'):
                current_faq['answer'] = line.split(':', 1)[1].strip()
        
        if 'question' in current_faq and 'answer' in current_faq:
            faqs.append(current_faq)
        
        return faqs

# Celery tasks for async processing
@celery_app.task(bind=True, max_retries=3)
def process_training_data_task(self, training_data_id: str, db_config: Dict[str, str]):
    """Celery task for processing training data"""
    try:
        service = EmbeddingService(db_config)
        result = asyncio.run(service.process_training_data(training_data_id))
        return result
    except Exception as e:
        logger.error(f"Training data processing task failed: {e}")
        self.retry(countdown=60 * (2 ** self.request.retries))

@celery_app.task(bind=True)
def generate_embeddings_task(self, texts: List[str], chatbot_id: str, source_type: str, 
                            db_config: Dict[str, str], **kwargs):
    """Celery task for generating embeddings"""
    try:
        service = EmbeddingService(db_config)
        request = EmbeddingRequest(
            texts=texts,
            chatbot_id=chatbot_id,
            source_type=source_type,
            **kwargs
        )
        
        result = asyncio.run(service.generate_embeddings(request))
        if result.success:
            embedding_ids = asyncio.run(service.store_embeddings(request, result))
            return {'success': True, 'embedding_ids': embedding_ids}
        else:
            return {'success': False, 'error': result.error_message}
    
    except Exception as e:
        logger.error(f"Embedding generation task failed: {e}")
        return {'success': False, 'error': str(e)}

# Usage example and testing
if __name__ == "__main__":
    import asyncio
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'chatflow_ai',
        'user': 'postgres',
        'password': 'your_password'
    }
    
    # Initialize service
    service = EmbeddingService(db_config)
    
    # Example: Process a training data record
    async def test_embedding_service():
        try:
            # Test basic embedding generation
            request = EmbeddingRequest(
                texts=["Hello, how can I help you?", "I need help with my account"],
                chatbot_id="223e4567-e89b-12d3-a456-426614174000",
                source_type="test"
            )
            
            result = await service.generate_embeddings(request)
            print(f"Embedding generation result: {result.success}")
            print(f"Embeddings count: {len(result.embeddings)}")
            print(f"Tokens used: {result.tokens_used}")
            print(f"Model used: {result.model_used}")
            
            if result.success:
                embedding_ids = await service.store_embeddings(request, result)
                print(f"Stored embeddings with IDs: {embedding_ids}")
        
        except Exception as e:
            logger.error(f"Test failed: {e}")
    
    # Run test
    # asyncio.run(test_embedding_service())