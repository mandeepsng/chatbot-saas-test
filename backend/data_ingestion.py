"""
Training Data Ingestion System for ChatBot SaaS
Handles file uploads, text extraction, and preprocessing
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import asyncpg
from celery import Celery
import aiofiles
import magic
from pathlib import Path

# Document processing
import pypdf2
import docx2txt
import chardet

# Data validation
from pydantic import BaseModel, validator
from sqlalchemy.dialects.postgresql import UUID

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Celery app configuration
celery_app = Celery(
    'data_ingestion',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

class ContentType(Enum):
    CONVERSATION = "conversation"
    FAQ = "faq"
    DOCUMENT = "document"
    KNOWLEDGE_BASE = "knowledge_base"
    CSV_DATA = "csv_data"

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TrainingDataItem:
    chatbot_id: str
    user_id: str
    content_type: ContentType
    raw_content: str
    title: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = None
    language: str = "en"
    source_file: Optional[str] = None
    metadata: Dict[str, Any] = None

class TrainingDataValidator(BaseModel):
    chatbot_id: str
    user_id: str
    content_type: str
    raw_content: str
    title: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = []
    language: str = "en"
    source_file: Optional[str] = None
    
    @validator('content_type')
    def validate_content_type(cls, v):
        if v not in [ct.value for ct in ContentType]:
            raise ValueError(f'Invalid content type: {v}')
        return v
    
    @validator('raw_content')
    def validate_content_length(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Content too short (minimum 10 characters)')
        if len(v) > 1000000:  # 1MB limit
            raise ValueError('Content too large (maximum 1MB)')
        return v

class DataIngestionService:
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.supported_formats = {
            'text/plain': self._process_text,
            'application/pdf': self._process_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._process_docx,
            'application/msword': self._process_doc,
            'text/csv': self._process_csv,
            'application/json': self._process_json,
            'application/vnd.ms-excel': self._process_excel,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': self._process_excel
        }
    
    async def get_db_connection(self):
        return await asyncpg.connect(**self.db_config)
    
    async def process_uploaded_file(
        self, 
        file_path: str, 
        chatbot_id: str, 
        user_id: str,
        content_type: ContentType,
        title: Optional[str] = None,
        category: Optional[str] = None,
        tags: List[str] = None
    ) -> Dict[str, Any]:
        """Process uploaded file and extract training data"""
        
        try:
            # Detect file type
            mime_type = magic.from_file(file_path, mime=True)
            
            if mime_type not in self.supported_formats:
                raise ValueError(f"Unsupported file type: {mime_type}")
            
            # Process file based on type
            processor = self.supported_formats[mime_type]
            extracted_data = await processor(file_path)
            
            # Create training data items
            training_items = []
            
            if isinstance(extracted_data, list):
                # Multiple items (e.g., CSV rows, JSON array)
                for i, item in enumerate(extracted_data):
                    training_item = TrainingDataItem(
                        chatbot_id=chatbot_id,
                        user_id=user_id,
                        content_type=content_type,
                        raw_content=item.get('content', str(item)),
                        title=item.get('title') or f"{title} - Part {i+1}" if title else None,
                        category=item.get('category') or category,
                        tags=item.get('tags') or tags or [],
                        source_file=os.path.basename(file_path),
                        metadata={'original_index': i, 'mime_type': mime_type}
                    )
                    training_items.append(training_item)
            else:
                # Single item (e.g., document text)
                training_item = TrainingDataItem(
                    chatbot_id=chatbot_id,
                    user_id=user_id,
                    content_type=content_type,
                    raw_content=extracted_data,
                    title=title,
                    category=category,
                    tags=tags or [],
                    source_file=os.path.basename(file_path),
                    metadata={'mime_type': mime_type}
                )
                training_items.append(training_item)
            
            # Queue for processing
            job_ids = []
            for item in training_items:
                job = process_training_data_async.delay(asdict(item))
                job_ids.append(job.id)
            
            return {
                'status': 'queued',
                'items_count': len(training_items),
                'job_ids': job_ids,
                'file_type': mime_type
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    
    async def _process_text(self, file_path: str) -> str:
        """Process plain text files"""
        async with aiofiles.open(file_path, 'rb') as f:
            raw_data = await f.read()
            
        # Detect encoding
        encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
        
        return raw_data.decode(encoding, errors='ignore')
    
    async def _process_pdf(self, file_path: str) -> str:
        """Process PDF files"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        return text.strip()
    
    async def _process_docx(self, file_path: str) -> str:
        """Process DOCX files"""
        return docx2txt.process(file_path)
    
    async def _process_doc(self, file_path: str) -> str:
        """Process legacy DOC files"""
        # Note: Would need python-docx or another library for full DOC support
        raise NotImplementedError("Legacy DOC format not fully supported. Convert to DOCX.")
    
    async def _process_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Process CSV files"""
        df = pd.read_csv(file_path)
        
        # Detect if it's Q&A format
        columns = df.columns.str.lower()
        
        if 'question' in columns and 'answer' in columns:
            # FAQ/Q&A format
            result = []
            for _, row in df.iterrows():
                result.append({
                    'content': f"Q: {row['question']}\nA: {row['answer']}",
                    'title': row.get('title', row['question'][:50]),
                    'category': row.get('category', 'FAQ'),
                    'tags': row.get('tags', '').split(',') if row.get('tags') else []
                })
            return result
        
        elif 'content' in columns:
            # Direct content format
            result = []
            for _, row in df.iterrows():
                result.append({
                    'content': row['content'],
                    'title': row.get('title', row['content'][:50]),
                    'category': row.get('category', 'Data'),
                    'tags': row.get('tags', '').split(',') if row.get('tags') else []
                })
            return result
        
        else:
            # Generic format - concatenate all columns
            result = []
            for _, row in df.iterrows():
                content = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                result.append({
                    'content': content,
                    'title': f"Row {_}",
                    'category': 'Data'
                })
            return result
    
    async def _process_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Process JSON files"""
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            data = json.loads(content)
        
        if isinstance(data, list):
            # Array of objects
            result = []
            for item in data:
                if isinstance(item, dict):
                    # Extract meaningful content
                    content = item.get('content') or item.get('text') or item.get('message') or str(item)
                    result.append({
                        'content': content,
                        'title': item.get('title') or item.get('subject') or content[:50],
                        'category': item.get('category') or item.get('type', 'JSON Data'),
                        'tags': item.get('tags', [])
                    })
                else:
                    result.append({'content': str(item), 'category': 'JSON Data'})
            return result
        
        elif isinstance(data, dict):
            # Single object
            content = data.get('content') or data.get('text') or json.dumps(data, indent=2)
            return [{
                'content': content,
                'title': data.get('title', 'JSON Document'),
                'category': data.get('category', 'JSON Data'),
                'tags': data.get('tags', [])
            }]
        
        else:
            # Primitive value
            return [{'content': str(data), 'category': 'JSON Data'}]
    
    async def _process_excel(self, file_path: str) -> List[Dict[str, Any]]:
        """Process Excel files"""
        df = pd.read_excel(file_path)
        return await self._process_dataframe_like_csv(df)
    
    async def _process_dataframe_like_csv(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Common processing logic for dataframe-like data"""
        columns = df.columns.str.lower()
        
        if 'question' in columns and 'answer' in columns:
            result = []
            for _, row in df.iterrows():
                result.append({
                    'content': f"Q: {row['question']}\nA: {row['answer']}",
                    'title': row.get('title', str(row['question'])[:50]),
                    'category': row.get('category', 'FAQ')
                })
            return result
        
        else:
            result = []
            for _, row in df.iterrows():
                content = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                result.append({
                    'content': content,
                    'title': f"Row {_}",
                    'category': 'Data'
                })
            return result

@celery_app.task(bind=True)
def process_training_data_async(self, training_data: Dict[str, Any]):
    """Async Celery task to process training data"""
    
    task_id = self.request.id
    
    try:
        # Validate data
        validator = TrainingDataValidator(**training_data)
        
        # Store in database
        result = asyncio.run(store_training_data(validator.dict()))
        
        # Trigger embedding generation
        from .embedding_service import generate_embeddings_async
        embedding_job = generate_embeddings_async.delay(result['training_data_id'])
        
        return {
            'status': 'completed',
            'training_data_id': result['training_data_id'],
            'embedding_job_id': embedding_job.id
        }
        
    except Exception as e:
        logger.error(f"Error in training data processing task {task_id}: {str(e)}")
        
        # Update status to failed
        asyncio.run(update_training_data_status(
            training_data.get('id'), 
            ProcessingStatus.FAILED.value, 
            str(e)
        ))
        
        raise

async def store_training_data(data: Dict[str, Any]) -> Dict[str, str]:
    """Store training data in database"""
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'chatbot_training'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    conn = await asyncpg.connect(**db_config)
    
    try:
        training_data_id = str(uuid.uuid4())
        
        await conn.execute("""
            INSERT INTO training_data (
                id, chatbot_id, user_id, content_type, source_file,
                raw_content, title, category, tags, language, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """, 
            training_data_id,
            data['chatbot_id'],
            data['user_id'],
            data['content_type'],
            data.get('source_file'),
            data['raw_content'],
            data.get('title'),
            data.get('category'),
            data.get('tags', []),
            data.get('language', 'en'),
            ProcessingStatus.COMPLETED.value
        )
        
        return {'training_data_id': training_data_id}
        
    finally:
        await conn.close()

async def update_training_data_status(
    training_data_id: str, 
    status: str, 
    error_message: Optional[str] = None
):
    """Update training data processing status"""
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'chatbot_training'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    conn = await asyncpg.connect(**db_config)
    
    try:
        await conn.execute("""
            UPDATE training_data 
            SET status = $1, processing_error = $2, updated_at = CURRENT_TIMESTAMP
            WHERE id = $3
        """, status, error_message, training_data_id)
        
    finally:
        await conn.close()

class ConversationExtractor:
    """Extract conversation pairs from chat logs or conversational data"""
    
    @staticmethod
    def extract_from_chat_log(chat_text: str) -> List[Dict[str, str]]:
        """Extract Q&A pairs from chat log text"""
        
        pairs = []
        lines = chat_text.strip().split('\n')
        
        current_user_msg = None
        current_bot_msg = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect user message (various formats)
            if (line.startswith('User:') or 
                line.startswith('Customer:') or 
                line.startswith('Q:') or
                line.startswith('>')):
                
                # Save previous pair
                if current_user_msg and current_bot_msg:
                    pairs.append({
                        'user_message': current_user_msg,
                        'bot_response': current_bot_msg
                    })
                
                current_user_msg = line.split(':', 1)[1].strip() if ':' in line else line[1:].strip()
                current_bot_msg = None
                
            # Detect bot message
            elif (line.startswith('Bot:') or 
                  line.startswith('Assistant:') or 
                  line.startswith('A:') or
                  line.startswith('<')):
                
                current_bot_msg = line.split(':', 1)[1].strip() if ':' in line else line[1:].strip()
        
        # Save final pair
        if current_user_msg and current_bot_msg:
            pairs.append({
                'user_message': current_user_msg,
                'bot_response': current_bot_msg
            })
        
        return pairs

# FastAPI endpoints for file upload integration
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta

app = FastAPI()

# Auth models
class UserSignup(BaseModel):
    email: str
    password: str
    company: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    company: str
    token: str

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ingestion_service = DataIngestionService({
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': os.getenv('DB_NAME', 'chatbot_saas_db'),
    'user': os.getenv('DB_USER', 'chatbot_saas_user'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
})

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "data-ingestion", "timestamp": datetime.utcnow()}

@app.post("/auth/signup")
async def signup(user: UserSignup):
    """User registration endpoint"""
    try:
        # Hash password
        hashed_password = pwd_context.hash(user.password)
        
        # Connect to database
        conn = await ingestion_service.get_db_connection()
        
        try:
            # Check if user exists
            existing = await conn.fetchrow(
                "SELECT id FROM users WHERE email = $1", user.email.lower()
            )
            
            if existing:
                raise HTTPException(status_code=400, detail="Email already registered")
            
            # Create user
            user_id = str(uuid.uuid4())
            await conn.execute("""
                INSERT INTO users (id, email, password_hash, company, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, user_id, user.email.lower(), hashed_password, user.company, 
                datetime.utcnow(), datetime.utcnow())
            
            # Create access token
            token = create_access_token({"user_id": user_id, "email": user.email})
            
            return {
                "message": "Account created successfully",
                "user": {
                    "id": user_id,
                    "email": user.email,
                    "company": user.company
                },
                "token": token
            }
            
        finally:
            await conn.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/auth/login")
async def login(user: UserLogin):
    """User login endpoint"""
    try:
        conn = await ingestion_service.get_db_connection()
        
        try:
            # Get user from database
            db_user = await conn.fetchrow("""
                SELECT id, email, password_hash, company
                FROM users 
                WHERE email = $1
            """, user.email.lower())
            
            if not db_user:
                raise HTTPException(status_code=401, detail="Invalid email or password")
            
            # Verify password
            if not pwd_context.verify(user.password, db_user['password_hash']):
                raise HTTPException(status_code=401, detail="Invalid email or password")
            
            # Create access token
            token = create_access_token({"user_id": db_user['id'], "email": db_user['email']})
            
            # Update last login
            await conn.execute(
                "UPDATE users SET last_login = $1 WHERE id = $2",
                datetime.utcnow(), db_user['id']
            )
            
            return {
                "message": "Login successful",
                "user": {
                    "id": db_user['id'],
                    "email": db_user['email'],
                    "company": db_user['company']
                },
                "token": token
            }
            
        finally:
            await conn.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/upload-training-data")
async def upload_training_data(
    file: UploadFile = File(...),
    chatbot_id: str = Form(...),
    user_id: str = Form(...),
    content_type: str = Form(...),
    title: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """Upload and process training data file"""
    
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
        
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Parse tags
        tag_list = [t.strip() for t in tags.split(',')] if tags else []
        
        # Process file
        result = await ingestion_service.process_uploaded_file(
            file_path=temp_path,
            chatbot_id=chatbot_id,
            user_id=user_id,
            content_type=ContentType(content_type),
            title=title,
            category=category,
            tags=tag_list
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return JSONResponse(content=result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/training-data/{chatbot_id}")
async def get_training_data(chatbot_id: str, page: int = 1, limit: int = 50):
    """Get training data for a chatbot"""
    
    conn = await ingestion_service.get_db_connection()
    
    try:
        offset = (page - 1) * limit
        
        rows = await conn.fetch("""
            SELECT id, content_type, title, category, tags, status, 
                   created_at, updated_at, LENGTH(raw_content) as content_length
            FROM training_data 
            WHERE chatbot_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
        """, chatbot_id, limit, offset)
        
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM training_data WHERE chatbot_id = $1",
            chatbot_id
        )
        
        return {
            'data': [dict(row) for row in rows],
            'total': total,
            'page': page,
            'limit': limit,
            'pages': (total + limit - 1) // limit
        }
        
    finally:
        await conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)