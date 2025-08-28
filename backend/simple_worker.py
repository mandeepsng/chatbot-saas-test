#!/usr/bin/env python3
"""
Simple background worker for ChatBot SaaS
Handles basic background tasks without Celery complexity
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, Any

import redis
import asyncpg
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleWorker:
    def __init__(self):
        self.redis_client = None
        self.db_pool = None
        self.running = True
        
    async def connect(self):
        """Connect to Redis and PostgreSQL"""
        try:
            # Redis connection
            redis_host = os.getenv('REDIS_HOST', 'redis')
            redis_port = int(os.getenv('REDIS_PORT', 6379))
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            
            # Test Redis connection
            self.redis_client.ping()
            logger.info("✅ Connected to Redis")
            
            # PostgreSQL connection
            db_host = os.getenv('POSTGRES_HOST', 'postgres')
            db_port = int(os.getenv('POSTGRES_PORT', 5432))
            db_name = os.getenv('POSTGRES_DB', 'chatbot_saas_db')
            db_user = os.getenv('POSTGRES_USER', 'chatbot_saas_user')
            db_password = os.getenv('POSTGRES_PASSWORD', 'password')
            
            self.db_pool = await asyncpg.create_pool(
                host=db_host,
                port=db_port,
                database=db_name,
                user=db_user,
                password=db_password,
                min_size=1,
                max_size=5
            )
            logger.info("✅ Connected to PostgreSQL")
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            raise
    
    async def process_embedding_task(self, task_data: Dict[str, Any]):
        """Process a simple embedding task"""
        try:
            text = task_data.get('text', '')
            user_id = task_data.get('user_id')
            
            # Simple embedding simulation (replace with actual embedding generation)
            embedding = [0.1] * 768  # Dummy embedding
            
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO training_embeddings (user_id, text_content, embedding, created_at)
                    VALUES ($1, $2, $3, $4)
                """, user_id, text, embedding, datetime.utcnow())
            
            logger.info(f"✅ Processed embedding task for user {user_id}")
            
        except Exception as e:
            logger.error(f"❌ Embedding task failed: {e}")
    
    async def process_training_task(self, task_data: Dict[str, Any]):
        """Process a simple training task"""
        try:
            user_id = task_data.get('user_id')
            
            # Simple training simulation
            await asyncio.sleep(2)  # Simulate processing time
            
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE chatbots SET 
                        status = 'trained',
                        updated_at = $1
                    WHERE user_id = $2 AND status = 'training'
                """, datetime.utcnow(), user_id)
            
            logger.info(f"✅ Processed training task for user {user_id}")
            
        except Exception as e:
            logger.error(f"❌ Training task failed: {e}")
    
    async def run(self):
        """Main worker loop"""
        logger.info("🚀 Simple Worker starting...")
        
        await self.connect()
        
        while self.running:
            try:
                # Check for embedding tasks
                embedding_task = self.redis_client.lpop('embedding_queue')
                if embedding_task:
                    task_data = json.loads(embedding_task)
                    await self.process_embedding_task(task_data)
                
                # Check for training tasks
                training_task = self.redis_client.lpop('training_queue')
                if training_task:
                    task_data = json.loads(training_task)
                    await self.process_training_task(task_data)
                
                # Health check update
                self.redis_client.setex('worker_health', 30, 'healthy')
                
                # Sleep if no tasks
                if not embedding_task and not training_task:
                    await asyncio.sleep(5)
                    
            except Exception as e:
                logger.error(f"❌ Worker error: {e}")
                await asyncio.sleep(10)  # Back off on error
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Worker shutting down...")
        self.running = False
        if self.db_pool:
            await self.db_pool.close()

async def main():
    worker = SimpleWorker()
    try:
        await worker.run()
    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal")
    finally:
        await worker.shutdown()

if __name__ == "__main__":
    asyncio.run(main())