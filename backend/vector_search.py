"""
Vector Similarity Search System for ChatBot SaaS
High-performance semantic search using PostgreSQL + pgvector
"""

import os
import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json

import asyncpg
import numpy as np
from pydantic import BaseModel, validator

# Search utilities
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    id: str
    similarity_score: float
    content: str
    source_type: str
    metadata: Dict[str, Any]
    chatbot_id: str

@dataclass
class SearchQuery:
    text: str
    chatbot_id: str
    max_results: int = 10
    similarity_threshold: float = 0.8
    filters: Optional[Dict[str, Any]] = None
    search_types: Optional[List[str]] = None  # question, answer, document_chunk, knowledge

class SearchRequest(BaseModel):
    query: str
    chatbot_id: str
    max_results: int = 10
    similarity_threshold: float = 0.8
    search_types: Optional[List[str]] = None
    category_filter: Optional[str] = None
    date_range: Optional[Dict[str, str]] = None
    
    @validator('similarity_threshold')
    def validate_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Similarity threshold must be between 0.0 and 1.0')
        return v
    
    @validator('max_results')
    def validate_max_results(cls, v):
        if not 1 <= v <= 100:
            raise ValueError('Max results must be between 1 and 100')
        return v

class VectorSearchService:
    def __init__(self, db_config: Dict[str, str], embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.db_config = db_config
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dimension = 384  # all-MiniLM-L6-v2 dimension
        
        # For OpenAI compatibility (if needed)
        self.openai_dimension = 1536
        
    async def get_db_connection(self):
        return await asyncpg.connect(**self.db_config)
    
    async def search_similar_content(
        self, 
        search_request: SearchRequest,
        user_id: Optional[str] = None
    ) -> List[SearchResult]:
        """Perform semantic similarity search across training data"""
        
        try:
            # Generate query embedding
            query_embedding = await self._generate_query_embedding(search_request.query)
            
            # Build search filters
            where_conditions, params = self._build_search_filters(
                search_request, 
                user_id, 
                len(query_embedding) + 1  # +1 for query_embedding param
            )
            
            # Execute vector search
            conn = await self.get_db_connection()
            
            try:
                # Set user context for RLS
                if user_id:
                    await conn.execute("SET app.user_id = $1", user_id)
                
                # Main similarity search query
                query = f"""
                SELECT 
                    te.id,
                    (1 - (te.embedding <=> $1))::float as similarity_score,
                    te.source_text,
                    te.source_type,
                    te.chatbot_id,
                    te.chunk_index,
                    te.usage_count,
                    te.last_used_at,
                    td.title,
                    td.category,
                    td.tags,
                    td.content_type,
                    cp.user_message,
                    cp.bot_response,
                    cp.intent,
                    cp.confidence_score
                FROM training_embeddings te
                LEFT JOIN training_data td ON te.training_data_id = td.id
                LEFT JOIN conversation_pairs cp ON te.conversation_pair_id = cp.id
                WHERE te.chatbot_id = $2
                AND (1 - (te.embedding <=> $1)) >= $3
                {where_conditions}
                ORDER BY te.embedding <=> $1
                LIMIT $4
                """
                
                rows = await conn.fetch(
                    query,
                    query_embedding,
                    search_request.chatbot_id,
                    search_request.similarity_threshold,
                    search_request.max_results,
                    *params
                )
                
                # Log search for analytics
                await self._log_search(
                    conn, 
                    search_request, 
                    query_embedding, 
                    len(rows),
                    user_id
                )
                
                # Update usage counts
                if rows:
                    embedding_ids = [row['id'] for row in rows]
                    await self._update_usage_counts(conn, embedding_ids)
                
                # Convert to SearchResult objects
                results = []
                for row in rows:
                    metadata = {
                        'title': row['title'],
                        'category': row['category'],
                        'tags': row['tags'] or [],
                        'content_type': row['content_type'],
                        'chunk_index': row['chunk_index'],
                        'usage_count': row['usage_count'],
                        'last_used_at': row['last_used_at'].isoformat() if row['last_used_at'] else None
                    }
                    
                    # Add conversation-specific metadata
                    if row['user_message']:
                        metadata.update({
                            'user_message': row['user_message'],
                            'bot_response': row['bot_response'],
                            'intent': row['intent'],
                            'confidence_score': row['confidence_score']
                        })
                    
                    results.append(SearchResult(
                        id=row['id'],
                        similarity_score=row['similarity_score'],
                        content=row['source_text'],
                        source_type=row['source_type'],
                        metadata=metadata,
                        chatbot_id=row['chatbot_id']
                    ))
                
                return results
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"Vector search error: {str(e)}")
            raise
    
    async def search_knowledge_base(
        self, 
        query: str, 
        chatbot_id: str, 
        max_results: int = 5,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search knowledge base entries"""
        
        query_embedding = await self._generate_query_embedding(query)
        
        conn = await self.get_db_connection()
        try:
            if user_id:
                await conn.execute("SET app.user_id = $1", user_id)
            
            rows = await conn.fetch("""
                SELECT 
                    id,
                    title,
                    content,
                    summary,
                    category,
                    tags,
                    keywords,
                    (1 - (embedding <=> $1))::float as similarity_score,
                    access_count,
                    effectiveness_score,
                    last_accessed_at
                FROM knowledge_base
                WHERE chatbot_id = $2
                AND (1 - (embedding <=> $1)) >= 0.7
                ORDER BY embedding <=> $1
                LIMIT $3
            """, query_embedding, chatbot_id, max_results)
            
            # Update access counts
            if rows:
                kb_ids = [row['id'] for row in rows]
                await conn.execute("""
                    UPDATE knowledge_base 
                    SET access_count = access_count + 1, 
                        last_accessed_at = CURRENT_TIMESTAMP
                    WHERE id = ANY($1)
                """, kb_ids)
            
            return [dict(row) for row in rows]
            
        finally:
            await conn.close()
    
    async def search_faq(
        self, 
        query: str, 
        chatbot_id: str, 
        confidence_threshold: float = 0.8,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Search FAQ entries for direct matches"""
        
        query_embedding = await self._generate_query_embedding(query)
        
        conn = await self.get_db_connection()
        try:
            if user_id:
                await conn.execute("SET app.user_id = $1", user_id)
            
            # Search both question and answer embeddings
            row = await conn.fetchrow("""
                WITH question_matches AS (
                    SELECT 
                        id, question, answer, category, priority,
                        (1 - (question_embedding <=> $1))::float as q_similarity,
                        'question' as match_type
                    FROM faq_entries
                    WHERE chatbot_id = $2 
                    AND is_active = true
                    AND (1 - (question_embedding <=> $1)) >= $3
                ),
                answer_matches AS (
                    SELECT 
                        id, question, answer, category, priority,
                        (1 - (answer_embedding <=> $1))::float as a_similarity,
                        'answer' as match_type
                    FROM faq_entries
                    WHERE chatbot_id = $2 
                    AND is_active = true
                    AND (1 - (answer_embedding <=> $1)) >= $3
                ),
                combined_matches AS (
                    SELECT *, q_similarity as similarity FROM question_matches
                    UNION ALL
                    SELECT *, a_similarity as similarity FROM answer_matches
                )
                SELECT *
                FROM combined_matches
                ORDER BY similarity DESC, priority DESC
                LIMIT 1
            """, query_embedding, chatbot_id, confidence_threshold)
            
            if row:
                # Update match statistics
                await conn.execute("""
                    UPDATE faq_entries 
                    SET match_count = match_count + 1,
                        last_matched_at = CURRENT_TIMESTAMP
                    WHERE id = $1
                """, row['id'])
                
                return dict(row)
            
            return None
            
        finally:
            await conn.close()
    
    async def hybrid_search(
        self, 
        query: str, 
        chatbot_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform hybrid search across all content types"""
        
        # Run searches in parallel
        tasks = [
            self.search_faq(query, chatbot_id, user_id=user_id),
            self.search_knowledge_base(query, chatbot_id, user_id=user_id),
            self.search_similar_content(
                SearchRequest(
                    query=query,
                    chatbot_id=chatbot_id,
                    max_results=10,
                    similarity_threshold=0.7
                ),
                user_id=user_id
            )
        ]
        
        faq_result, kb_results, similarity_results = await asyncio.gather(*tasks)
        
        return {
            'faq_match': faq_result,
            'knowledge_base': kb_results[:3],  # Top 3 KB entries
            'similar_content': similarity_results[:5],  # Top 5 similar content
            'total_results': len(kb_results) + len(similarity_results) + (1 if faq_result else 0)
        }
    
    async def contextual_search(
        self, 
        query: str, 
        conversation_history: List[Dict[str, str]], 
        chatbot_id: str,
        user_id: Optional[str] = None
    ) -> List[SearchResult]:
        """Search with conversation context"""
        
        # Build contextual query
        context_window = self._build_context_window(conversation_history, query)
        contextual_embedding = await self._generate_query_embedding(context_window)
        
        # Search with context
        conn = await self.get_db_connection()
        try:
            if user_id:
                await conn.execute("SET app.user_id = $1", user_id)
            
            # First try context embeddings
            context_rows = await conn.fetch("""
                SELECT 
                    ce.id,
                    ce.context_window,
                    (1 - (ce.embedding <=> $1))::float as similarity_score
                FROM context_embeddings ce
                WHERE ce.chatbot_id = $2
                AND (1 - (ce.embedding <=> $1)) >= 0.75
                ORDER BY ce.embedding <=> $1
                LIMIT 5
            """, contextual_embedding, chatbot_id)
            
            # Then search training embeddings with context boost
            training_rows = await conn.fetch("""
                SELECT 
                    te.id,
                    (1 - (te.embedding <=> $1))::float as similarity_score,
                    te.source_text,
                    te.source_type,
                    te.chatbot_id
                FROM training_embeddings te
                WHERE te.chatbot_id = $2
                AND (1 - (te.embedding <=> $1)) >= 0.7
                ORDER BY te.embedding <=> $1
                LIMIT 10
            """, contextual_embedding, chatbot_id)
            
            # Combine and rank results
            results = []
            
            # Context-aware results get priority
            for row in context_rows:
                results.append(SearchResult(
                    id=row['id'],
                    similarity_score=row['similarity_score'] * 1.1,  # Context boost
                    content=row['context_window'],
                    source_type='context',
                    metadata={'source': 'conversation_context'},
                    chatbot_id=chatbot_id
                ))
            
            # Add training results
            for row in training_rows:
                results.append(SearchResult(
                    id=row['id'],
                    similarity_score=row['similarity_score'],
                    content=row['source_text'],
                    source_type=row['source_type'],
                    metadata={'source': 'training_data'},
                    chatbot_id=row['chatbot_id']
                ))
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:8]
            
        finally:
            await conn.close()
    
    def _build_context_window(
        self, 
        conversation_history: List[Dict[str, str]], 
        current_query: str
    ) -> str:
        """Build context window from conversation history"""
        
        context_parts = []
        
        # Add recent conversation turns (last 3-4 exchanges)
        recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        
        for turn in recent_history:
            if turn.get('user'):
                context_parts.append(f"User: {turn['user']}")
            if turn.get('assistant'):
                context_parts.append(f"Assistant: {turn['assistant']}")
        
        # Add current query
        context_parts.append(f"Current Query: {current_query}")
        
        return "\n".join(context_parts)
    
    async def _generate_query_embedding(self, text: str) -> List[float]:
        """Generate embedding for search query"""
        
        # Use local model for consistent results
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        
        # Convert to list for PostgreSQL
        return embedding.tolist()
    
    def _build_search_filters(
        self, 
        search_request: SearchRequest, 
        user_id: Optional[str],
        param_offset: int
    ) -> Tuple[str, List[Any]]:
        """Build WHERE clause and parameters for search filters"""
        
        conditions = []
        params = []
        param_count = param_offset
        
        # Source type filter
        if search_request.search_types:
            param_count += 1
            conditions.append(f"AND te.source_type = ANY(${param_count})")
            params.append(search_request.search_types)
        
        # Category filter
        if search_request.category_filter:
            param_count += 1
            conditions.append(f"AND td.category = ${param_count}")
            params.append(search_request.category_filter)
        
        # Date range filter
        if search_request.date_range:
            if search_request.date_range.get('start'):
                param_count += 1
                conditions.append(f"AND te.created_at >= ${param_count}")
                params.append(search_request.date_range['start'])
            
            if search_request.date_range.get('end'):
                param_count += 1
                conditions.append(f"AND te.created_at <= ${param_count}")
                params.append(search_request.date_range['end'])
        
        where_clause = " ".join(conditions)
        return where_clause, params
    
    async def _log_search(
        self, 
        conn: asyncpg.Connection, 
        search_request: SearchRequest,
        query_embedding: List[float],
        results_count: int,
        user_id: Optional[str]
    ):
        """Log search query for analytics"""
        
        try:
            await conn.execute("""
                INSERT INTO similarity_search_logs (
                    chatbot_id, query_text, query_embedding, 
                    similarity_threshold, max_results, results_count,
                    user_id, response_time_ms
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                search_request.chatbot_id,
                search_request.query,
                query_embedding,
                search_request.similarity_threshold,
                search_request.max_results,
                results_count,
                user_id,
                50  # Placeholder for response time
            )
        except Exception as e:
            logger.warning(f"Failed to log search: {str(e)}")
    
    async def _update_usage_counts(self, conn: asyncpg.Connection, embedding_ids: List[str]):
        """Update usage statistics for retrieved embeddings"""
        
        try:
            await conn.execute("""
                UPDATE training_embeddings 
                SET usage_count = usage_count + 1,
                    last_used_at = CURRENT_TIMESTAMP
                WHERE id = ANY($1)
            """, embedding_ids)
        except Exception as e:
            logger.warning(f"Failed to update usage counts: {str(e)}")

# Search analytics and optimization
class SearchAnalytics:
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
    
    async def get_search_metrics(self, chatbot_id: str, days: int = 30) -> Dict[str, Any]:
        """Get search analytics for a chatbot"""
        
        conn = await asyncpg.connect(**self.db_config)
        try:
            # Search volume and patterns
            metrics = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_searches,
                    AVG(results_count) as avg_results_per_search,
                    AVG(response_time_ms) as avg_response_time,
                    COUNT(DISTINCT user_id) as unique_users
                FROM similarity_search_logs
                WHERE chatbot_id = $1
                AND created_at >= CURRENT_DATE - INTERVAL '%s days'
            """, chatbot_id, days)
            
            # Popular queries
            popular_queries = await conn.fetch("""
                SELECT query_text, COUNT(*) as frequency
                FROM similarity_search_logs
                WHERE chatbot_id = $1
                AND created_at >= CURRENT_DATE - INTERVAL '%s days'
                GROUP BY query_text
                ORDER BY frequency DESC
                LIMIT 10
            """, chatbot_id, days)
            
            # Low-performing searches (no results or low similarity)
            low_performance = await conn.fetch("""
                SELECT query_text, AVG(top_similarity_score) as avg_similarity
                FROM similarity_search_logs
                WHERE chatbot_id = $1
                AND created_at >= CURRENT_DATE - INTERVAL '%s days'
                AND (results_count = 0 OR top_similarity_score < 0.7)
                GROUP BY query_text
                ORDER BY avg_similarity ASC
                LIMIT 10
            """, chatbot_id, days)
            
            return {
                'metrics': dict(metrics) if metrics else {},
                'popular_queries': [dict(row) for row in popular_queries],
                'low_performance_queries': [dict(row) for row in low_performance]
            }
            
        finally:
            await conn.close()

# FastAPI endpoints
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

search_app = FastAPI()
search_service = VectorSearchService({
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': os.getenv('DB_NAME', 'chatbot_training'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
})

@search_app.post("/api/search")
async def search_content(request: SearchRequest, user_id: str = Query(None)):
    """Semantic search endpoint"""
    
    try:
        results = await search_service.search_similar_content(request, user_id)
        
        return {
            'results': [
                {
                    'id': r.id,
                    'content': r.content,
                    'similarity_score': r.similarity_score,
                    'source_type': r.source_type,
                    'metadata': r.metadata
                }
                for r in results
            ],
            'total': len(results),
            'query': request.query,
            'threshold': request.similarity_threshold
        }
        
    except Exception as e:
        logger.error(f"Search API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Search failed")

@search_app.get("/api/search/hybrid/{chatbot_id}")
async def hybrid_search_endpoint(
    chatbot_id: str, 
    q: str = Query(..., description="Search query"),
    user_id: str = Query(None)
):
    """Hybrid search across all content types"""
    
    try:
        results = await search_service.hybrid_search(q, chatbot_id, user_id)
        return results
        
    except Exception as e:
        logger.error(f"Hybrid search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Hybrid search failed")

@search_app.get("/api/search/analytics/{chatbot_id}")
async def get_search_analytics(chatbot_id: str, days: int = 30):
    """Get search analytics"""
    
    try:
        analytics = SearchAnalytics(search_service.db_config)
        metrics = await analytics.get_search_metrics(chatbot_id, days)
        return metrics
        
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        raise HTTPException(status_code=500, detail="Analytics retrieval failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(search_app, host="0.0.0.0", port=8002)