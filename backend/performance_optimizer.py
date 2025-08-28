"""
Performance Optimization System for Vector Database Training
Handles caching, indexing, query optimization, and resource management
"""

import os
import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import hashlib
from dataclasses import dataclass
from enum import Enum

import asyncpg
import redis
import numpy as np
from celery import Celery
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Monitoring
import psutil
import GPUtil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    INDEX_OPTIMIZATION = "index_optimization"
    CACHE_WARMING = "cache_warming"
    VECTOR_CLUSTERING = "vector_clustering"
    QUERY_OPTIMIZATION = "query_optimization"
    RESOURCE_SCALING = "resource_scaling"

@dataclass
class PerformanceMetrics:
    avg_query_time: float
    cache_hit_rate: float
    index_usage_rate: float
    memory_usage_percent: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float]
    active_connections: int
    queries_per_second: float

class VectorCache:
    """High-performance caching for vector embeddings and search results"""
    
    def __init__(self, redis_client: redis.Redis, ttl_seconds: int = 3600):
        self.redis = redis_client
        self.ttl = ttl_seconds
        
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key"""
        if isinstance(data, dict):
            key_data = json.dumps(data, sort_keys=True)
        else:
            key_data = str(data)
        
        hash_key = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{hash_key}"
    
    async def get_embedding(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get cached embedding"""
        key = self._generate_key("embedding", {"text": text, "model": model_name})
        
        try:
            cached = await self.redis.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        
        return None
    
    async def set_embedding(self, text: str, model_name: str, embedding: List[float]):
        """Cache embedding"""
        key = self._generate_key("embedding", {"text": text, "model": model_name})
        
        try:
            await self.redis.setex(
                key, 
                self.ttl, 
                json.dumps(embedding)
            )
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    async def get_search_results(
        self, 
        query: str, 
        chatbot_id: str, 
        params: Dict[str, Any]
    ) -> Optional[List[Dict]]:
        """Get cached search results"""
        key = self._generate_key("search", {
            "query": query,
            "chatbot_id": chatbot_id,
            "params": params
        })
        
        try:
            cached = await self.redis.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Search cache get error: {e}")
        
        return None
    
    async def set_search_results(
        self,
        query: str,
        chatbot_id: str,
        params: Dict[str, Any],
        results: List[Dict]
    ):
        """Cache search results"""
        key = self._generate_key("search", {
            "query": query,
            "chatbot_id": chatbot_id,
            "params": params
        })
        
        try:
            # Shorter TTL for search results (they may become stale)
            await self.redis.setex(
                key,
                min(self.ttl, 1800),  # Max 30 minutes
                json.dumps(results)
            )
        except Exception as e:
            logger.warning(f"Search cache set error: {e}")
    
    async def invalidate_chatbot_cache(self, chatbot_id: str):
        """Invalidate all cache entries for a chatbot"""
        try:
            pattern = f"*{chatbot_id}*"
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries for chatbot {chatbot_id}")
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")

class IndexOptimizer:
    """Optimizes database indexes for vector operations"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
    
    async def get_db_connection(self):
        return await asyncpg.connect(**self.db_config)
    
    async def analyze_index_usage(self, chatbot_id: str) -> Dict[str, Any]:
        """Analyze index usage patterns"""
        
        conn = await self.get_db_connection()
        try:
            # Get index statistics
            stats = await conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes
                WHERE tablename IN (
                    'training_embeddings', 'context_embeddings', 
                    'knowledge_base', 'faq_entries'
                )
                ORDER BY idx_scan DESC
            """)
            
            # Get table sizes
            sizes = await conn.fetch("""
                SELECT 
                    tablename,
                    pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size,
                    pg_total_relation_size(tablename::regclass) as size_bytes
                FROM pg_tables
                WHERE tablename IN (
                    'training_embeddings', 'context_embeddings',
                    'knowledge_base', 'faq_entries'
                )
            """)
            
            # Check for unused indexes
            unused_indexes = await conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname
                FROM pg_stat_user_indexes
                WHERE idx_scan = 0
                AND tablename IN (
                    'training_embeddings', 'context_embeddings',
                    'knowledge_base', 'faq_entries'
                )
            """)
            
            return {
                'index_stats': [dict(row) for row in stats],
                'table_sizes': [dict(row) for row in sizes],
                'unused_indexes': [dict(row) for row in unused_indexes]
            }
            
        finally:
            await conn.close()
    
    async def optimize_vector_indexes(self, chatbot_id: str) -> Dict[str, Any]:
        """Optimize vector indexes for better performance"""
        
        conn = await self.get_db_connection()
        try:
            optimizations = []
            
            # Check if we need to rebuild HNSW indexes
            index_conditions = await conn.fetch("""
                SELECT 
                    tablename,
                    COUNT(*) as row_count,
                    AVG(LENGTH(embedding::text)) as avg_embedding_size
                FROM (
                    SELECT 'training_embeddings' as tablename, embedding FROM training_embeddings WHERE chatbot_id = $1
                    UNION ALL
                    SELECT 'knowledge_base' as tablename, embedding FROM knowledge_base WHERE chatbot_id = $1
                    UNION ALL
                    SELECT 'faq_entries' as tablename, question_embedding as embedding FROM faq_entries WHERE chatbot_id = $1
                ) combined
                GROUP BY tablename
            """, chatbot_id)
            
            for condition in index_conditions:
                table = condition['tablename']
                row_count = condition['row_count']
                
                # Rebuild index if table has grown significantly
                if row_count > 10000:
                    try:
                        if table == 'training_embeddings':
                            await conn.execute("REINDEX INDEX idx_training_embeddings_vector")
                        elif table == 'knowledge_base':
                            await conn.execute("REINDEX INDEX idx_knowledge_base_vector")
                        elif table == 'faq_entries':
                            await conn.execute("REINDEX INDEX idx_faq_question_vector")
                        
                        optimizations.append(f"Rebuilt HNSW index for {table}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to rebuild index for {table}: {e}")
                
                # Create partial indexes for frequently accessed data
                if row_count > 50000:
                    try:
                        partial_index_name = f"idx_{table}_recent_vectors"
                        
                        if table == 'training_embeddings':
                            await conn.execute(f"""
                                CREATE INDEX CONCURRENTLY IF NOT EXISTS {partial_index_name}
                                ON training_embeddings USING hnsw (embedding vector_cosine_ops)
                                WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
                                AND usage_count > 0
                            """)
                        
                        optimizations.append(f"Created partial index {partial_index_name}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to create partial index for {table}: {e}")
            
            # Update table statistics
            tables_to_analyze = ['training_embeddings', 'knowledge_base', 'faq_entries', 'context_embeddings']
            for table in tables_to_analyze:
                await conn.execute(f"ANALYZE {table}")
            
            optimizations.append("Updated table statistics")
            
            return {
                'optimizations_applied': optimizations,
                'table_conditions': [dict(row) for row in index_conditions]
            }
            
        finally:
            await conn.close()

class VectorClusterOptimizer:
    """Optimizes vector storage through clustering and dimensionality reduction"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
    
    async def get_db_connection(self):
        return await asyncpg.connect(**self.db_config)
    
    async def analyze_vector_clusters(self, chatbot_id: str) -> Dict[str, Any]:
        """Analyze vector distribution and clustering patterns"""
        
        conn = await self.get_db_connection()
        try:
            # Sample vectors for analysis (limit to avoid memory issues)
            vectors = await conn.fetch("""
                SELECT embedding, source_type, usage_count
                FROM training_embeddings
                WHERE chatbot_id = $1
                AND embedding IS NOT NULL
                ORDER BY RANDOM()
                LIMIT 1000
            """, chatbot_id)
            
            if len(vectors) < 50:
                return {'error': 'Insufficient vectors for clustering analysis'}
            
            # Convert to numpy array
            vector_data = np.array([list(v['embedding']) for v in vectors])
            source_types = [v['source_type'] for v in vectors]
            usage_counts = [v['usage_count'] or 0 for v in vectors]
            
            # Perform clustering analysis
            optimal_k = min(10, len(vectors) // 10)  # Rule of thumb
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            clusters = kmeans.fit_predict(vector_data)
            
            # Analyze clusters
            cluster_analysis = {}
            for i in range(optimal_k):
                cluster_mask = clusters == i
                cluster_analysis[f'cluster_{i}'] = {
                    'size': int(np.sum(cluster_mask)),
                    'avg_usage': float(np.mean(np.array(usage_counts)[cluster_mask])),
                    'dominant_source_type': max(
                        set(np.array(source_types)[cluster_mask]),
                        key=lambda x: list(np.array(source_types)[cluster_mask]).count(x)
                    ) if np.sum(cluster_mask) > 0 else 'unknown'
                }
            
            # Calculate inertia (within-cluster sum of squares)
            inertia = float(kmeans.inertia_)
            
            # PCA for dimensionality analysis
            pca = PCA()
            pca.fit(vector_data)
            explained_variance = pca.explained_variance_ratio_
            
            # Determine optimal dimensions (95% variance)
            cumsum = np.cumsum(explained_variance)
            optimal_dims = int(np.argmax(cumsum >= 0.95) + 1)
            
            return {
                'total_vectors': len(vectors),
                'optimal_clusters': optimal_k,
                'cluster_analysis': cluster_analysis,
                'clustering_inertia': inertia,
                'original_dimensions': len(vector_data[0]),
                'optimal_dimensions': optimal_dims,
                'explained_variance_ratio': explained_variance[:10].tolist(),  # Top 10 components
                'vector_statistics': {
                    'mean_usage_count': float(np.mean(usage_counts)),
                    'max_usage_count': int(np.max(usage_counts)),
                    'source_type_distribution': {
                        st: source_types.count(st) for st in set(source_types)
                    }
                }
            }
            
        finally:
            await conn.close()
    
    async def create_vector_clusters(self, chatbot_id: str, num_clusters: int = 5):
        """Create optimized vector clusters for faster similarity search"""
        
        conn = await self.get_db_connection()
        try:
            # Create cluster metadata table if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_clusters (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    chatbot_id UUID NOT NULL,
                    cluster_id INTEGER NOT NULL,
                    centroid vector(1536),
                    cluster_size INTEGER DEFAULT 0,
                    avg_similarity_threshold FLOAT DEFAULT 0.8,
                    dominant_source_type VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(chatbot_id, cluster_id)
                )
            """)
            
            # Get all vectors for clustering
            vectors = await conn.fetch("""
                SELECT id, embedding, source_type
                FROM training_embeddings
                WHERE chatbot_id = $1
                AND embedding IS NOT NULL
            """, chatbot_id)
            
            if len(vectors) < num_clusters * 2:
                raise ValueError(f"Insufficient vectors for {num_clusters} clusters")
            
            # Perform clustering
            vector_data = np.array([list(v['embedding']) for v in vectors])
            vector_ids = [v['id'] for v in vectors]
            source_types = [v['source_type'] for v in vectors]
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(vector_data)
            centroids = kmeans.cluster_centers_
            
            # Store cluster information
            await conn.execute("DELETE FROM vector_clusters WHERE chatbot_id = $1", chatbot_id)
            
            for i, centroid in enumerate(centroids):
                cluster_mask = cluster_labels == i
                cluster_vectors = np.array(vector_ids)[cluster_mask]
                cluster_sources = np.array(source_types)[cluster_mask]
                
                # Find dominant source type
                dominant_source = max(
                    set(cluster_sources),
                    key=lambda x: list(cluster_sources).count(x)
                ) if len(cluster_sources) > 0 else 'unknown'
                
                # Calculate average similarity within cluster
                if np.sum(cluster_mask) > 1:
                    cluster_data = vector_data[cluster_mask]
                    similarities = []
                    for vec in cluster_data:
                        sim = np.dot(vec, centroid) / (np.linalg.norm(vec) * np.linalg.norm(centroid))
                        similarities.append(sim)
                    avg_similarity = float(np.mean(similarities))
                else:
                    avg_similarity = 1.0
                
                await conn.execute("""
                    INSERT INTO vector_clusters (
                        chatbot_id, cluster_id, centroid, cluster_size,
                        avg_similarity_threshold, dominant_source_type
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    chatbot_id,
                    i,
                    centroid.tolist(),
                    int(np.sum(cluster_mask)),
                    avg_similarity,
                    dominant_source
                )
            
            # Add cluster assignments to embeddings
            await conn.execute("""
                ALTER TABLE training_embeddings 
                ADD COLUMN IF NOT EXISTS cluster_id INTEGER
            """)
            
            # Update cluster assignments
            for i, vector_id in enumerate(vector_ids):
                await conn.execute("""
                    UPDATE training_embeddings 
                    SET cluster_id = $1 
                    WHERE id = $2
                """, int(cluster_labels[i]), vector_id)
            
            return {
                'clusters_created': num_clusters,
                'vectors_clustered': len(vectors),
                'cluster_sizes': [int(np.sum(cluster_labels == i)) for i in range(num_clusters)]
            }
            
        finally:
            await conn.close()

class ResourceMonitor:
    """Monitors system resources and suggests optimizations"""
    
    @staticmethod
    def get_system_metrics() -> PerformanceMetrics:
        """Get current system performance metrics"""
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU (if available)
        gpu_percent = None
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].load * 100
        except:
            pass
        
        return PerformanceMetrics(
            avg_query_time=0.0,  # Placeholder - would be calculated from logs
            cache_hit_rate=0.0,  # Placeholder - would come from cache stats
            index_usage_rate=0.0,  # Placeholder - would come from DB stats
            memory_usage_percent=memory.percent,
            cpu_usage_percent=cpu_percent,
            gpu_usage_percent=gpu_percent,
            active_connections=0,  # Placeholder - would come from DB
            queries_per_second=0.0  # Placeholder - would be calculated
        )
    
    @staticmethod
    async def get_database_metrics(db_config: Dict[str, str]) -> Dict[str, Any]:
        """Get database performance metrics"""
        
        conn = await asyncpg.connect(**db_config)
        try:
            # Connection stats
            connections = await conn.fetchrow("""
                SELECT 
                    count(*) as total_connections,
                    count(*) FILTER (WHERE state = 'active') as active_connections,
                    count(*) FILTER (WHERE state = 'idle') as idle_connections
                FROM pg_stat_activity
            """)
            
            # Query performance
            slow_queries = await conn.fetch("""
                SELECT 
                    query,
                    calls,
                    total_time,
                    mean_time,
                    rows
                FROM pg_stat_statements
                WHERE mean_time > 100  -- Queries taking more than 100ms on average
                ORDER BY mean_time DESC
                LIMIT 10
            """)
            
            # Database size
            db_size = await conn.fetchrow("""
                SELECT 
                    pg_size_pretty(pg_database_size(current_database())) as db_size,
                    pg_database_size(current_database()) as db_size_bytes
            """)
            
            # Index hit ratio
            index_hit_ratio = await conn.fetchrow("""
                SELECT 
                    round(
                        (sum(idx_blks_hit) / nullif(sum(idx_blks_hit + idx_blks_read), 0)) * 100,
                        2
                    ) as index_hit_ratio
                FROM pg_statio_user_indexes
            """)
            
            return {
                'connections': dict(connections) if connections else {},
                'slow_queries': [dict(row) for row in slow_queries],
                'database_size': dict(db_size) if db_size else {},
                'index_hit_ratio': index_hit_ratio['index_hit_ratio'] if index_hit_ratio else 0
            }
            
        finally:
            await conn.close()

class PerformanceOptimizer:
    """Main performance optimization orchestrator"""
    
    def __init__(self, db_config: Dict[str, str], redis_client: redis.Redis):
        self.db_config = db_config
        self.cache = VectorCache(redis_client)
        self.index_optimizer = IndexOptimizer(db_config)
        self.cluster_optimizer = VectorClusterOptimizer(db_config)
        self.monitor = ResourceMonitor()
    
    async def run_optimization_suite(self, chatbot_id: str) -> Dict[str, Any]:
        """Run complete optimization suite"""
        
        results = {}
        
        try:
            # 1. Analyze current performance
            results['system_metrics'] = self.monitor.get_system_metrics()
            results['database_metrics'] = await self.monitor.get_database_metrics(self.db_config)
            
            # 2. Optimize indexes
            results['index_optimization'] = await self.index_optimizer.optimize_vector_indexes(chatbot_id)
            
            # 3. Analyze vector clusters
            results['cluster_analysis'] = await self.cluster_optimizer.analyze_vector_clusters(chatbot_id)
            
            # 4. Create optimized clusters if beneficial
            cluster_analysis = results['cluster_analysis']
            if (not cluster_analysis.get('error') and 
                cluster_analysis.get('total_vectors', 0) > 100):
                
                optimal_clusters = min(cluster_analysis.get('optimal_clusters', 5), 10)
                results['cluster_optimization'] = await self.cluster_optimizer.create_vector_clusters(
                    chatbot_id, 
                    optimal_clusters
                )
            
            # 5. Warm up cache with frequently accessed vectors
            await self._warm_cache(chatbot_id)
            results['cache_warming'] = {'status': 'completed'}
            
            # 6. Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Optimization suite error: {str(e)}")
            return {'error': str(e)}
    
    async def _warm_cache(self, chatbot_id: str):
        """Warm up cache with frequently accessed data"""
        
        conn = await asyncpg.connect(**self.db_config)
        try:
            # Get most frequently used embeddings
            frequent_embeddings = await conn.fetch("""
                SELECT embedding, source_text, usage_count
                FROM training_embeddings
                WHERE chatbot_id = $1
                AND usage_count > 5
                ORDER BY usage_count DESC, last_used_at DESC
                LIMIT 100
            """, chatbot_id)
            
            # Cache them (this would be done by the actual search/embedding service)
            logger.info(f"Would cache {len(frequent_embeddings)} frequently used embeddings")
            
        finally:
            await conn.close()
    
    def _generate_recommendations(self, optimization_results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on analysis"""
        
        recommendations = []
        
        # System resource recommendations
        system_metrics = optimization_results.get('system_metrics')
        if system_metrics:
            if system_metrics.memory_usage_percent > 80:
                recommendations.append("Consider increasing system memory - currently at {:.1f}%".format(
                    system_metrics.memory_usage_percent
                ))
            
            if system_metrics.cpu_usage_percent > 70:
                recommendations.append("High CPU usage detected - consider scaling to more cores")
        
        # Database recommendations
        db_metrics = optimization_results.get('database_metrics')
        if db_metrics:
            index_hit_ratio = db_metrics.get('index_hit_ratio', 0)
            if index_hit_ratio < 90:
                recommendations.append(f"Low index hit ratio ({index_hit_ratio}%) - consider index optimization")
            
            slow_queries = db_metrics.get('slow_queries', [])
            if slow_queries:
                recommendations.append(f"Found {len(slow_queries)} slow queries - review query optimization")
        
        # Vector clustering recommendations
        cluster_analysis = optimization_results.get('cluster_analysis')
        if cluster_analysis and not cluster_analysis.get('error'):
            optimal_dims = cluster_analysis.get('optimal_dimensions', 0)
            original_dims = cluster_analysis.get('original_dimensions', 0)
            
            if optimal_dims < original_dims * 0.8:
                recommendations.append(f"Consider dimensionality reduction: {optimal_dims} vs {original_dims} dimensions")
        
        # Index optimization recommendations
        index_opt = optimization_results.get('index_optimization')
        if index_opt:
            optimizations = index_opt.get('optimizations_applied', [])
            if optimizations:
                recommendations.append(f"Applied {len(optimizations)} index optimizations")
        
        return recommendations

# Celery app for background optimization
celery_app = Celery(
    'performance_optimizer',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

@celery_app.task
def optimize_chatbot_performance(chatbot_id: str):
    """Background task to optimize chatbot performance"""
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'chatbot_training'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
    
    optimizer = PerformanceOptimizer(db_config, redis_client)
    
    result = asyncio.run(optimizer.run_optimization_suite(chatbot_id))
    
    return result

# FastAPI endpoints
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse

perf_app = FastAPI()

@perf_app.post("/api/optimize/{chatbot_id}")
async def optimize_performance(chatbot_id: str, background_tasks: BackgroundTasks):
    """Start performance optimization"""
    
    task = optimize_chatbot_performance.delay(chatbot_id)
    
    return {
        'task_id': task.id,
        'status': 'started',
        'chatbot_id': chatbot_id
    }

@perf_app.get("/api/metrics/system")
async def get_system_metrics():
    """Get current system metrics"""
    
    monitor = ResourceMonitor()
    metrics = monitor.get_system_metrics()
    
    return {
        'cpu_usage': metrics.cpu_usage_percent,
        'memory_usage': metrics.memory_usage_percent,
        'gpu_usage': metrics.gpu_usage_percent
    }

@perf_app.get("/api/metrics/database")
async def get_database_metrics():
    """Get database performance metrics"""
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'chatbot_training'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    monitor = ResourceMonitor()
    metrics = await monitor.get_database_metrics(db_config)
    
    return metrics

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(perf_app, host="0.0.0.0", port=8004)