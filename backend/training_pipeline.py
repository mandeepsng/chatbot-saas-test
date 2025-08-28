"""
Model Training Pipeline for ChatBot SaaS
Orchestrates the complete training process from data preparation to model deployment
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import asyncpg
import numpy as np
from celery import Celery, group, chord
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# ML libraries
import torch
from transformers import (
    AutoTokenizer, AutoModel, 
    TrainingArguments, Trainer,
    AutoModelForSequenceClassification
)
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Celery configuration
celery_app = Celery(
    'training_pipeline',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

class TrainingJobType(Enum):
    INITIAL_TRAINING = "initial_training"
    INCREMENTAL = "incremental"
    RETRAINING = "retraining"
    FINE_TUNING = "fine_tuning"

class TrainingStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TrainingConfig:
    model_type: str = "embedding_based"  # embedding_based, fine_tuned, hybrid
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_length: int = 512
    validation_split: float = 0.2
    similarity_threshold: float = 0.8
    use_gpu: bool = True
    early_stopping: bool = True
    save_best_model: bool = True

@dataclass
class TrainingJob:
    id: str
    chatbot_id: str
    user_id: str
    job_type: TrainingJobType
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.QUEUED
    progress: float = 0.0
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class TrainingPipeline:
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.model_storage_path = "/app/models"
        os.makedirs(self.model_storage_path, exist_ok=True)
        
    async def get_db_connection(self):
        return await asyncpg.connect(**self.db_config)
    
    async def start_training_job(
        self, 
        chatbot_id: str, 
        user_id: str, 
        job_type: TrainingJobType,
        config: TrainingConfig
    ) -> str:
        """Start a new training job"""
        
        job_id = str(uuid.uuid4())
        
        # Create job record
        conn = await self.get_db_connection()
        try:
            await conn.execute("""
                INSERT INTO training_jobs (
                    id, chatbot_id, user_id, job_type, status,
                    model_config, training_params, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                job_id,
                chatbot_id,
                user_id,
                job_type.value,
                TrainingStatus.QUEUED.value,
                json.dumps(asdict(config)),
                json.dumps({}),
                datetime.utcnow()
            )
            
            # Queue training task
            train_model_async.delay(job_id, chatbot_id, asdict(config))
            
            return job_id
            
        finally:
            await conn.close()
    
    async def get_training_data(self, chatbot_id: str) -> Tuple[List[Dict], List[Dict]]:
        """Retrieve and prepare training data"""
        
        conn = await self.get_db_connection()
        try:
            # Get conversation pairs for Q&A training
            conversation_data = await conn.fetch("""
                SELECT 
                    cp.user_message,
                    cp.bot_response,
                    cp.context,
                    cp.intent,
                    cp.confidence_score,
                    td.category,
                    td.tags
                FROM conversation_pairs cp
                JOIN training_data td ON cp.training_data_id = td.id
                WHERE cp.chatbot_id = $1
                AND td.status = 'completed'
                AND LENGTH(cp.user_message) > 10
                AND LENGTH(cp.bot_response) > 10
            """, chatbot_id)
            
            # Get general training data for context learning
            training_data = await conn.fetch("""
                SELECT 
                    processed_content,
                    title,
                    category,
                    tags,
                    content_type
                FROM training_data
                WHERE chatbot_id = $1
                AND status = 'completed'
                AND processed_content IS NOT NULL
                AND LENGTH(processed_content) > 50
            """, chatbot_id)
            
            return [dict(row) for row in conversation_data], [dict(row) for row in training_data]
            
        finally:
            await conn.close()
    
    async def prepare_training_examples(
        self, 
        conversation_data: List[Dict],
        training_data: List[Dict],
        config: TrainingConfig
    ) -> List[InputExample]:
        """Prepare training examples for sentence transformers"""
        
        examples = []
        
        # Positive examples from Q&A pairs
        for i, conv in enumerate(conversation_data):
            # Question-Answer similarity
            examples.append(InputExample(
                texts=[conv['user_message'], conv['bot_response']],
                label=1.0
            ))
            
            # Add context if available
            if conv['context']:
                examples.append(InputExample(
                    texts=[conv['context'], conv['user_message']],
                    label=0.8
                ))
        
        # Generate negative examples (non-matching Q&A pairs)
        if len(conversation_data) > 1:
            for i, conv1 in enumerate(conversation_data[:100]):  # Limit for performance
                for j, conv2 in enumerate(conversation_data[i+1:i+6]):  # Limited negative sampling
                    if conv1['intent'] != conv2['intent']:  # Different intents
                        examples.append(InputExample(
                            texts=[conv1['user_message'], conv2['bot_response']],
                            label=0.2
                        ))
        
        # Add training data similarity examples
        for i, data in enumerate(training_data[:50]):  # Limit for performance
            if data['title'] and data['processed_content']:
                examples.append(InputExample(
                    texts=[data['title'], data['processed_content']],
                    label=0.9
                ))
        
        logger.info(f"Prepared {len(examples)} training examples")
        return examples
    
    async def train_embedding_model(
        self,
        job_id: str,
        chatbot_id: str,
        examples: List[InputExample],
        config: TrainingConfig
    ) -> str:
        """Train custom embedding model"""
        
        try:
            # Load base model
            model = SentenceTransformer(config.base_model)
            
            # Create output directory
            model_dir = os.path.join(self.model_storage_path, chatbot_id, job_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # Split data
            train_examples, val_examples = train_test_split(
                examples, 
                test_size=config.validation_split,
                random_state=42
            )
            
            # Create data loader
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=config.batch_size)
            
            # Define loss function
            train_loss = losses.CosineSimilarityLoss(model)
            
            # Training arguments
            warmup_steps = int(len(train_dataloader) * 0.1)
            
            # Update progress
            await self._update_job_progress(job_id, 20, "Starting model training...")
            
            # Train the model
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=config.num_epochs,
                warmup_steps=warmup_steps,
                output_path=model_dir,
                save_best_model=config.save_best_model,
                show_progress_bar=True
            )
            
            # Update progress
            await self._update_job_progress(job_id, 80, "Evaluating model...")
            
            # Evaluate model
            metrics = await self._evaluate_model(model, val_examples)
            
            # Save model metadata
            model_info = {
                'model_path': model_dir,
                'base_model': config.base_model,
                'training_examples': len(examples),
                'validation_examples': len(val_examples),
                'metrics': metrics,
                'config': asdict(config),
                'trained_at': datetime.utcnow().isoformat()
            }
            
            model_info_path = os.path.join(model_dir, 'model_info.json')
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            return model_dir
            
        except Exception as e:
            logger.error(f"Model training error: {str(e)}")
            raise
    
    async def _evaluate_model(
        self, 
        model: SentenceTransformer, 
        val_examples: List[InputExample]
    ) -> Dict[str, float]:
        """Evaluate trained model"""
        
        if len(val_examples) < 10:
            return {'note': 'Insufficient validation data for evaluation'}
        
        # Prepare validation data
        sentences1 = [ex.texts[0] for ex in val_examples]
        sentences2 = [ex.texts[1] for ex in val_examples]
        labels = [ex.label for ex in val_examples]
        
        # Generate embeddings
        embeddings1 = model.encode(sentences1)
        embeddings2 = model.encode(sentences2)
        
        # Calculate cosine similarities
        similarities = [
            np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            for e1, e2 in zip(embeddings1, embeddings2)
        ]
        
        # Binary classification metrics (threshold = 0.5)
        predictions = [1 if sim > 0.5 else 0 for sim in similarities]
        binary_labels = [1 if label > 0.5 else 0 for label in labels]
        
        accuracy = accuracy_score(binary_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            binary_labels, predictions, average='binary'
        )
        
        # Correlation with actual similarity scores
        correlation = np.corrcoef(similarities, labels)[0, 1]
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'similarity_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'avg_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities))
        }
    
    async def create_model_version(
        self,
        job_id: str,
        chatbot_id: str,
        model_path: str,
        metrics: Dict[str, float]
    ) -> str:
        """Create new model version record"""
        
        conn = await self.get_db_connection()
        try:
            # Get current version number
            current_version = await conn.fetchval("""
                SELECT COALESCE(MAX(CAST(SUBSTRING(version_number FROM '^v(.*)') AS INTEGER)), 0)
                FROM model_versions
                WHERE chatbot_id = $1
            """, chatbot_id)
            
            new_version = f"v{current_version + 1}"
            version_id = str(uuid.uuid4())
            
            # Deactivate current active version
            await conn.execute("""
                UPDATE model_versions 
                SET is_active = false
                WHERE chatbot_id = $1 AND is_active = true
            """, chatbot_id)
            
            # Create new version
            await conn.execute("""
                INSERT INTO model_versions (
                    id, chatbot_id, training_job_id, version_number,
                    model_path, is_active, accuracy_score, f1_score,
                    response_time_ms, deployed_at, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
                version_id,
                chatbot_id,
                job_id,
                new_version,
                model_path,
                True,
                metrics.get('accuracy'),
                metrics.get('f1_score'),
                50,  # Estimated response time
                datetime.utcnow(),
                datetime.utcnow()
            )
            
            return version_id
            
        finally:
            await conn.close()
    
    async def _update_job_progress(
        self, 
        job_id: str, 
        progress: float, 
        status_message: str = None
    ):
        """Update job progress"""
        
        conn = await self.get_db_connection()
        try:
            await conn.execute("""
                UPDATE training_jobs 
                SET progress_percentage = $1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = $2
            """, progress, job_id)
            
            if status_message:
                logger.info(f"Job {job_id}: {status_message} ({progress}%)")
                
        except Exception as e:
            logger.warning(f"Failed to update job progress: {str(e)}")
        finally:
            await conn.close()
    
    async def _update_job_status(
        self,
        job_id: str,
        status: TrainingStatus,
        error_message: str = None,
        metrics: Dict[str, float] = None
    ):
        """Update job status"""
        
        conn = await self.get_db_connection()
        try:
            end_time = datetime.utcnow() if status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED] else None
            
            await conn.execute("""
                UPDATE training_jobs 
                SET status = $1,
                    error_message = $2,
                    model_metrics = $3,
                    completed_at = $4,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = $5
            """, 
                status.value,
                error_message,
                json.dumps(metrics) if metrics else None,
                end_time,
                job_id
            )
            
        finally:
            await conn.close()

# Celery tasks
@celery_app.task(bind=True)
def train_model_async(self, job_id: str, chatbot_id: str, config_dict: Dict):
    """Async model training task"""
    
    task_id = self.request.id
    logger.info(f"Starting training job {job_id} (task: {task_id})")
    
    try:
        config = TrainingConfig(**config_dict)
        pipeline = TrainingPipeline({
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'chatbot_training'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres')
        })
        
        # Run training pipeline
        result = asyncio.run(_run_training_pipeline(pipeline, job_id, chatbot_id, config))
        
        return result
        
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {str(e)}")
        
        # Update job status to failed
        pipeline = TrainingPipeline({
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'chatbot_training'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres')
        })
        
        asyncio.run(pipeline._update_job_status(
            job_id, 
            TrainingStatus.FAILED, 
            str(e)
        ))
        
        raise

async def _run_training_pipeline(
    pipeline: TrainingPipeline,
    job_id: str,
    chatbot_id: str,
    config: TrainingConfig
) -> Dict[str, Any]:
    """Run the complete training pipeline"""
    
    try:
        # Update status to processing
        await pipeline._update_job_status(job_id, TrainingStatus.PROCESSING)
        await pipeline._update_job_progress(job_id, 5, "Loading training data...")
        
        # Get training data
        conversation_data, training_data = await pipeline.get_training_data(chatbot_id)
        
        if len(conversation_data) == 0 and len(training_data) == 0:
            raise ValueError("No training data available for this chatbot")
        
        await pipeline._update_job_progress(job_id, 15, "Preparing training examples...")
        
        # Prepare examples
        examples = await pipeline.prepare_training_examples(
            conversation_data, training_data, config
        )
        
        if len(examples) < 10:
            raise ValueError("Insufficient training examples (minimum 10 required)")
        
        # Train model
        await pipeline._update_job_progress(job_id, 25, "Training embedding model...")
        model_path = await pipeline.train_embedding_model(job_id, chatbot_id, examples, config)
        
        await pipeline._update_job_progress(job_id, 90, "Creating model version...")
        
        # Load metrics from saved model info
        model_info_path = os.path.join(model_path, 'model_info.json')
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        metrics = model_info['metrics']
        
        # Create model version
        version_id = await pipeline.create_model_version(job_id, chatbot_id, model_path, metrics)
        
        # Update job status to completed
        await pipeline._update_job_status(job_id, TrainingStatus.COMPLETED, metrics=metrics)
        await pipeline._update_job_progress(job_id, 100, "Training completed successfully!")
        
        # Update chatbot last trained timestamp
        conn = await pipeline.get_db_connection()
        try:
            await conn.execute("""
                UPDATE chatbots 
                SET last_trained_at = CURRENT_TIMESTAMP,
                    status = 'active',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = $1
            """, chatbot_id)
        finally:
            await conn.close()
        
        return {
            'status': 'completed',
            'model_path': model_path,
            'version_id': version_id,
            'metrics': metrics,
            'training_examples': len(examples)
        }
        
    except Exception as e:
        await pipeline._update_job_status(job_id, TrainingStatus.FAILED, str(e))
        raise

# Training management API
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

class TrainingRequest(BaseModel):
    chatbot_id: str
    job_type: str = "initial_training"
    config: Optional[Dict[str, Any]] = None

training_app = FastAPI()
pipeline = TrainingPipeline({
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': os.getenv('DB_NAME', 'chatbot_training'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
})

@training_app.post("/api/training/start")
async def start_training(request: TrainingRequest, user_id: str):
    """Start a new training job"""
    
    try:
        # Parse config
        config_dict = request.config or {}
        config = TrainingConfig(**config_dict)
        
        # Start training job
        job_id = await pipeline.start_training_job(
            request.chatbot_id,
            user_id,
            TrainingJobType(request.job_type),
            config
        )
        
        return {'job_id': job_id, 'status': 'queued'}
        
    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@training_app.get("/api/training/status/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status"""
    
    conn = await pipeline.get_db_connection()
    try:
        job = await conn.fetchrow("""
            SELECT * FROM training_jobs WHERE id = $1
        """, job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return dict(job)
        
    finally:
        await conn.close()

@training_app.get("/api/training/jobs/{chatbot_id}")
async def get_training_jobs(chatbot_id: str, limit: int = 20):
    """Get training jobs for a chatbot"""
    
    conn = await pipeline.get_db_connection()
    try:
        jobs = await conn.fetch("""
            SELECT * FROM training_jobs 
            WHERE chatbot_id = $1
            ORDER BY created_at DESC
            LIMIT $2
        """, chatbot_id, limit)
        
        return [dict(job) for job in jobs]
        
    finally:
        await conn.close()

@training_app.get("/api/models/{chatbot_id}")
async def get_model_versions(chatbot_id: str):
    """Get model versions for a chatbot"""
    
    conn = await pipeline.get_db_connection()
    try:
        versions = await conn.fetch("""
            SELECT * FROM model_versions 
            WHERE chatbot_id = $1
            ORDER BY created_at DESC
        """, chatbot_id)
        
        return [dict(version) for version in versions]
        
    finally:
        await conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(training_app, host="0.0.0.0", port=8003)