"""
Monitoring and Analytics System for Vector Database Training
Comprehensive monitoring, logging, and analytics for the chatbot training system
"""

import os
import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import asyncpg
import numpy as np
from celery import Celery
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Monitoring libraries
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
EMBEDDING_GENERATION_TIME = Histogram('embedding_generation_seconds', 'Time spent generating embeddings')
SEARCH_REQUEST_TIME = Histogram('search_request_seconds', 'Time spent on search requests')
TRAINING_JOB_DURATION = Histogram('training_job_seconds', 'Training job duration')

ACTIVE_CHATBOTS = Gauge('active_chatbots_total', 'Number of active chatbots')
TOTAL_EMBEDDINGS = Gauge('embeddings_total', 'Total number of embeddings stored')
SEARCH_REQUESTS = Counter('search_requests_total', 'Total search requests', ['chatbot_id', 'status'])
TRAINING_JOBS = Counter('training_jobs_total', 'Training jobs', ['status', 'job_type'])

DATABASE_CONNECTIONS = Gauge('database_connections_active', 'Active database connections')
CACHE_HIT_RATE = Gauge('cache_hit_rate', 'Cache hit rate percentage')

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    id: str
    chatbot_id: str
    alert_type: str
    level: AlertLevel
    title: str
    description: str
    threshold_value: float
    current_value: float
    created_at: datetime
    resolved_at: Optional[datetime] = None

class MetricsCollector:
    """Collects and aggregates system metrics"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        
    async def get_db_connection(self):
        return await asyncpg.connect(**self.db_config)
    
    async def collect_training_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Collect training-related metrics"""
        
        conn = await self.get_db_connection()
        try:
            # Training job statistics
            job_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_jobs,
                    COUNT(*) FILTER (WHERE status = 'completed') as completed_jobs,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed_jobs,
                    COUNT(*) FILTER (WHERE status = 'processing') as running_jobs,
                    AVG(processing_time_seconds) FILTER (WHERE status = 'completed') as avg_duration,
                    MAX(processing_time_seconds) FILTER (WHERE status = 'completed') as max_duration
                FROM training_jobs
                WHERE created_at >= CURRENT_DATE - INTERVAL '%s days'
            """, days)
            
            # Training data statistics
            data_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_training_data,
                    COUNT(*) FILTER (WHERE status = 'completed') as processed_data,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed_data,
                    AVG(LENGTH(raw_content)) as avg_content_length,
                    COUNT(DISTINCT chatbot_id) as active_chatbots
                FROM training_data
                WHERE created_at >= CURRENT_DATE - INTERVAL '%s days'
            """, days)
            
            # Embedding statistics
            embedding_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_embeddings,
                    AVG(usage_count) as avg_usage_count,
                    MAX(usage_count) as max_usage_count,
                    COUNT(DISTINCT chatbot_id) as chatbots_with_embeddings
                FROM training_embeddings
                WHERE created_at >= CURRENT_DATE - INTERVAL '%s days'
            """, days)
            
            # Model performance trends
            model_performance = await conn.fetch("""
                SELECT 
                    DATE(created_at) as date,
                    AVG(accuracy_score) as avg_accuracy,
                    AVG(f1_score) as avg_f1_score,
                    COUNT(*) as models_created
                FROM model_versions
                WHERE created_at >= CURRENT_DATE - INTERVAL '%s days'
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            """, days)
            
            return {
                'job_statistics': dict(job_stats) if job_stats else {},
                'data_statistics': dict(data_stats) if data_stats else {},
                'embedding_statistics': dict(embedding_stats) if embedding_stats else {},
                'model_performance_trends': [dict(row) for row in model_performance],
                'collection_period_days': days
            }
            
        finally:
            await conn.close()
    
    async def collect_search_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Collect search-related metrics"""
        
        conn = await self.get_db_connection()
        try:
            # Search volume and performance
            search_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_searches,
                    COUNT(DISTINCT chatbot_id) as active_chatbots,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(results_count) as avg_results_per_search,
                    AVG(response_time_ms) as avg_response_time,
                    AVG(top_similarity_score) as avg_similarity_score
                FROM similarity_search_logs
                WHERE created_at >= CURRENT_DATE - INTERVAL '%s days'
            """, days)
            
            # Search patterns by hour
            hourly_patterns = await conn.fetch("""
                SELECT 
                    EXTRACT(HOUR FROM created_at) as hour,
                    COUNT(*) as search_count,
                    AVG(response_time_ms) as avg_response_time
                FROM similarity_search_logs
                WHERE created_at >= CURRENT_DATE - INTERVAL '%s days'
                GROUP BY EXTRACT(HOUR FROM created_at)
                ORDER BY hour
            """, days)
            
            # Top performing queries
            top_queries = await conn.fetch("""
                SELECT 
                    query_text,
                    COUNT(*) as frequency,
                    AVG(top_similarity_score) as avg_similarity,
                    AVG(results_count) as avg_results
                FROM similarity_search_logs
                WHERE created_at >= CURRENT_DATE - INTERVAL '%s days'
                AND LENGTH(query_text) > 3
                GROUP BY query_text
                HAVING COUNT(*) > 1
                ORDER BY frequency DESC
                LIMIT 20
            """, days)
            
            # Low-performing queries (potential gaps)
            low_performance_queries = await conn.fetch("""
                SELECT 
                    query_text,
                    COUNT(*) as frequency,
                    AVG(top_similarity_score) as avg_similarity,
                    AVG(results_count) as avg_results
                FROM similarity_search_logs
                WHERE created_at >= CURRENT_DATE - INTERVAL '%s days'
                AND (results_count = 0 OR top_similarity_score < 0.6)
                GROUP BY query_text
                HAVING COUNT(*) > 2
                ORDER BY frequency DESC
                LIMIT 10
            """, days)
            
            return {
                'search_statistics': dict(search_stats) if search_stats else {},
                'hourly_patterns': [dict(row) for row in hourly_patterns],
                'top_queries': [dict(row) for row in top_queries],
                'low_performance_queries': [dict(row) for row in low_performance_queries],
                'collection_period_days': days
            }
            
        finally:
            await conn.close()
    
    async def collect_quality_metrics(self, chatbot_id: str) -> Dict[str, Any]:
        """Collect data quality metrics for a specific chatbot"""
        
        conn = await self.get_db_connection()
        try:
            # Data quality scores
            quality_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_assessments,
                    AVG(completeness_score) as avg_completeness,
                    AVG(consistency_score) as avg_consistency,
                    AVG(diversity_score) as avg_diversity,
                    AVG(relevance_score) as avg_relevance
                FROM data_quality_metrics
                WHERE chatbot_id = $1
            """, chatbot_id)
            
            # Recent quality trends
            quality_trends = await conn.fetch("""
                SELECT 
                    DATE(analyzed_at) as date,
                    AVG(completeness_score) as completeness,
                    AVG(consistency_score) as consistency,
                    AVG(diversity_score) as diversity,
                    AVG(relevance_score) as relevance
                FROM data_quality_metrics
                WHERE chatbot_id = $1
                AND analyzed_at >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY DATE(analyzed_at)
                ORDER BY date DESC
            """, chatbot_id)
            
            # Quality issues summary
            quality_issues = await conn.fetch("""
                SELECT 
                    issue->'type' as issue_type,
                    COUNT(*) as frequency
                FROM data_quality_metrics,
                     jsonb_array_elements(quality_issues) as issue
                WHERE chatbot_id = $1
                GROUP BY issue->'type'
                ORDER BY frequency DESC
            """, chatbot_id)
            
            return {
                'quality_statistics': dict(quality_stats) if quality_stats else {},
                'quality_trends': [dict(row) for row in quality_trends],
                'quality_issues': [dict(row) for row in quality_issues]
            }
            
        finally:
            await conn.close()

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.alert_thresholds = {
            'search_response_time': 1000,  # ms
            'training_failure_rate': 0.2,  # 20%
            'cache_hit_rate': 0.8,  # 80%
            'embedding_generation_time': 30,  # seconds
            'low_similarity_rate': 0.3  # 30%
        }
    
    async def get_db_connection(self):
        return await asyncpg.connect(**self.db_config)
    
    async def check_system_health(self) -> List[Alert]:
        """Check system health and generate alerts"""
        
        alerts = []
        
        # Check training job failure rate
        training_alert = await self._check_training_failure_rate()
        if training_alert:
            alerts.append(training_alert)
        
        # Check search performance
        search_alert = await self._check_search_performance()
        if search_alert:
            alerts.append(search_alert)
        
        # Check data quality issues
        quality_alerts = await self._check_data_quality_issues()
        alerts.extend(quality_alerts)
        
        # Store alerts
        for alert in alerts:
            await self._store_alert(alert)
        
        return alerts
    
    async def _check_training_failure_rate(self) -> Optional[Alert]:
        """Check if training failure rate is too high"""
        
        conn = await self.get_db_connection()
        try:
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_jobs,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed_jobs
                FROM training_jobs
                WHERE created_at >= CURRENT_DATE - INTERVAL '24 hours'
            """)
            
            if stats and stats['total_jobs'] > 0:
                failure_rate = stats['failed_jobs'] / stats['total_jobs']
                threshold = self.alert_thresholds['training_failure_rate']
                
                if failure_rate > threshold:
                    return Alert(
                        id=str(uuid.uuid4()),
                        chatbot_id="system",
                        alert_type="training_failure_rate",
                        level=AlertLevel.WARNING if failure_rate < threshold * 1.5 else AlertLevel.ERROR,
                        title="High Training Job Failure Rate",
                        description=f"Training job failure rate is {failure_rate:.1%}, above threshold of {threshold:.1%}",
                        threshold_value=threshold,
                        current_value=failure_rate,
                        created_at=datetime.utcnow()
                    )
            
        finally:
            await conn.close()
        
        return None
    
    async def _check_search_performance(self) -> Optional[Alert]:
        """Check search response time performance"""
        
        conn = await self.get_db_connection()
        try:
            stats = await conn.fetchrow("""
                SELECT 
                    AVG(response_time_ms) as avg_response_time,
                    COUNT(*) as total_searches
                FROM similarity_search_logs
                WHERE created_at >= CURRENT_DATE - INTERVAL '1 hour'
            """)
            
            if stats and stats['total_searches'] > 10:
                avg_time = stats['avg_response_time']
                threshold = self.alert_thresholds['search_response_time']
                
                if avg_time > threshold:
                    return Alert(
                        id=str(uuid.uuid4()),
                        chatbot_id="system",
                        alert_type="search_performance",
                        level=AlertLevel.WARNING if avg_time < threshold * 1.5 else AlertLevel.ERROR,
                        title="Slow Search Response Time",
                        description=f"Average search response time is {avg_time:.0f}ms, above threshold of {threshold}ms",
                        threshold_value=threshold,
                        current_value=avg_time,
                        created_at=datetime.utcnow()
                    )
            
        finally:
            await conn.close()
        
        return None
    
    async def _check_data_quality_issues(self) -> List[Alert]:
        """Check for data quality issues across chatbots"""
        
        alerts = []
        conn = await self.get_db_connection()
        
        try:
            # Get chatbots with poor data quality
            poor_quality = await conn.fetch("""
                SELECT 
                    chatbot_id,
                    AVG(completeness_score) as avg_completeness,
                    AVG(consistency_score) as avg_consistency,
                    AVG(diversity_score) as avg_diversity,
                    COUNT(*) as assessment_count
                FROM data_quality_metrics
                WHERE analyzed_at >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY chatbot_id
                HAVING (
                    AVG(completeness_score) < 0.7 OR
                    AVG(consistency_score) < 0.7 OR
                    AVG(diversity_score) < 0.5
                )
                AND COUNT(*) > 3
            """)
            
            for chatbot in poor_quality:
                chatbot_id = chatbot['chatbot_id']
                
                # Identify specific issues
                issues = []
                if chatbot['avg_completeness'] < 0.7:
                    issues.append(f"Low completeness score: {chatbot['avg_completeness']:.2f}")
                if chatbot['avg_consistency'] < 0.7:
                    issues.append(f"Low consistency score: {chatbot['avg_consistency']:.2f}")
                if chatbot['avg_diversity'] < 0.5:
                    issues.append(f"Low diversity score: {chatbot['avg_diversity']:.2f}")
                
                alert = Alert(
                    id=str(uuid.uuid4()),
                    chatbot_id=chatbot_id,
                    alert_type="data_quality",
                    level=AlertLevel.WARNING,
                    title="Data Quality Issues Detected",
                    description=f"Quality issues found: {'; '.join(issues)}",
                    threshold_value=0.7,
                    current_value=min(
                        chatbot['avg_completeness'],
                        chatbot['avg_consistency'],
                        chatbot['avg_diversity']
                    ),
                    created_at=datetime.utcnow()
                )
                alerts.append(alert)
            
        finally:
            await conn.close()
        
        return alerts
    
    async def _store_alert(self, alert: Alert):
        """Store alert in database"""
        
        conn = await self.get_db_connection()
        try:
            # Create alerts table if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS system_alerts (
                    id UUID PRIMARY KEY,
                    chatbot_id VARCHAR(255),
                    alert_type VARCHAR(100),
                    level VARCHAR(20),
                    title VARCHAR(500),
                    description TEXT,
                    threshold_value FLOAT,
                    current_value FLOAT,
                    created_at TIMESTAMP,
                    resolved_at TIMESTAMP,
                    acknowledged BOOLEAN DEFAULT false
                )
            """)
            
            await conn.execute("""
                INSERT INTO system_alerts (
                    id, chatbot_id, alert_type, level, title, description,
                    threshold_value, current_value, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (id) DO NOTHING
            """,
                alert.id,
                alert.chatbot_id,
                alert.alert_type,
                alert.level.value,
                alert.title,
                alert.description,
                alert.threshold_value,
                alert.current_value,
                alert.created_at
            )
            
        finally:
            await conn.close()

class AnalyticsDashboard:
    """Generates analytics and insights for the dashboard"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.metrics_collector = MetricsCollector(db_config)
    
    async def generate_chatbot_analytics(self, chatbot_id: str) -> Dict[str, Any]:
        """Generate comprehensive analytics for a chatbot"""
        
        # Collect different types of metrics
        training_metrics = await self.metrics_collector.collect_training_metrics(30)
        search_metrics = await self.metrics_collector.collect_search_metrics(30)
        quality_metrics = await self.metrics_collector.collect_quality_metrics(chatbot_id)
        
        # Generate insights
        insights = await self._generate_insights(chatbot_id, training_metrics, search_metrics, quality_metrics)
        
        # Calculate performance scores
        performance_scores = self._calculate_performance_scores(training_metrics, search_metrics, quality_metrics)
        
        return {
            'chatbot_id': chatbot_id,
            'training_metrics': training_metrics,
            'search_metrics': search_metrics,
            'quality_metrics': quality_metrics,
            'performance_scores': performance_scores,
            'insights': insights,
            'generated_at': datetime.utcnow().isoformat()
        }
    
    async def _generate_insights(
        self, 
        chatbot_id: str,
        training_metrics: Dict,
        search_metrics: Dict,
        quality_metrics: Dict
    ) -> List[str]:
        """Generate actionable insights"""
        
        insights = []
        
        # Training insights
        job_stats = training_metrics.get('job_statistics', {})
        if job_stats.get('total_jobs', 0) > 0:
            success_rate = job_stats.get('completed_jobs', 0) / job_stats['total_jobs']
            if success_rate < 0.9:
                insights.append(f"Training success rate is {success_rate:.1%} - investigate failed jobs")
            
            avg_duration = job_stats.get('avg_duration', 0)
            if avg_duration > 1800:  # 30 minutes
                insights.append(f"Average training time is {avg_duration/60:.1f} minutes - consider optimization")
        
        # Search insights
        search_stats = search_metrics.get('search_statistics', {})
        if search_stats.get('total_searches', 0) > 0:
            avg_results = search_stats.get('avg_results_per_search', 0)
            if avg_results < 3:
                insights.append(f"Low search results ({avg_results:.1f} avg) - may need more training data")
            
            avg_similarity = search_stats.get('avg_similarity_score', 0)
            if avg_similarity < 0.7:
                insights.append(f"Low similarity scores ({avg_similarity:.2f} avg) - consider model retraining")
        
        # Quality insights
        quality_stats = quality_metrics.get('quality_statistics', {})
        if quality_stats:
            if quality_stats.get('avg_completeness', 1) < 0.8:
                insights.append("Data completeness is low - review training data quality")
            
            if quality_stats.get('avg_diversity', 1) < 0.6:
                insights.append("Training data lacks diversity - add varied content types")
        
        return insights
    
    def _calculate_performance_scores(
        self,
        training_metrics: Dict,
        search_metrics: Dict,
        quality_metrics: Dict
    ) -> Dict[str, float]:
        """Calculate overall performance scores"""
        
        scores = {}
        
        # Training performance score
        job_stats = training_metrics.get('job_statistics', {})
        if job_stats.get('total_jobs', 0) > 0:
            success_rate = job_stats.get('completed_jobs', 0) / job_stats['total_jobs']
            speed_score = min(1.0, 1800 / max(job_stats.get('avg_duration', 1800), 1))  # Normalize to 30min
            scores['training_performance'] = (success_rate * 0.7 + speed_score * 0.3) * 100
        else:
            scores['training_performance'] = 0
        
        # Search performance score
        search_stats = search_metrics.get('search_statistics', {})
        if search_stats.get('total_searches', 0) > 0:
            response_time = search_stats.get('avg_response_time', 1000)
            speed_score = min(1.0, 100 / max(response_time, 1))  # Normalize to 100ms
            
            similarity_score = search_stats.get('avg_similarity_score', 0.5)
            results_score = min(1.0, search_stats.get('avg_results_per_search', 0) / 5)  # Normalize to 5 results
            
            scores['search_performance'] = (speed_score * 0.3 + similarity_score * 0.5 + results_score * 0.2) * 100
        else:
            scores['search_performance'] = 0
        
        # Data quality score
        quality_stats = quality_metrics.get('quality_statistics', {})
        if quality_stats:
            completeness = quality_stats.get('avg_completeness', 0.5)
            consistency = quality_stats.get('avg_consistency', 0.5)
            diversity = quality_stats.get('avg_diversity', 0.5)
            relevance = quality_stats.get('avg_relevance', 0.5)
            
            scores['data_quality'] = (completeness * 0.3 + consistency * 0.3 + diversity * 0.2 + relevance * 0.2) * 100
        else:
            scores['data_quality'] = 0
        
        # Overall score
        valid_scores = [s for s in scores.values() if s > 0]
        scores['overall'] = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        
        return scores

# Background monitoring tasks
celery_app = Celery(
    'monitoring_analytics',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

@celery_app.task
def collect_system_metrics():
    """Periodic task to collect system metrics"""
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'chatbot_training'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    collector = MetricsCollector(db_config)
    
    # Update Prometheus metrics
    training_metrics = asyncio.run(collector.collect_training_metrics(1))
    search_metrics = asyncio.run(collector.collect_search_metrics(1))
    
    # Update gauges
    training_stats = training_metrics.get('job_statistics', {})
    ACTIVE_CHATBOTS.set(training_stats.get('active_chatbots', 0))
    
    embedding_stats = training_metrics.get('embedding_statistics', {})
    TOTAL_EMBEDDINGS.set(embedding_stats.get('total_embeddings', 0))
    
    return {
        'training_metrics': training_metrics,
        'search_metrics': search_metrics,
        'timestamp': datetime.utcnow().isoformat()
    }

@celery_app.task
def check_system_alerts():
    """Periodic task to check for system alerts"""
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'chatbot_training'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    alert_manager = AlertManager(db_config)
    alerts = asyncio.run(alert_manager.check_system_health())
    
    return {
        'alerts_generated': len(alerts),
        'alerts': [asdict(alert) for alert in alerts],
        'timestamp': datetime.utcnow().isoformat()
    }

# FastAPI endpoints for monitoring
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

monitor_app = FastAPI()

@monitor_app.get("/api/analytics/{chatbot_id}")
async def get_chatbot_analytics(chatbot_id: str):
    """Get comprehensive analytics for a chatbot"""
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'chatbot_training'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    dashboard = AnalyticsDashboard(db_config)
    analytics = await dashboard.generate_chatbot_analytics(chatbot_id)
    
    return analytics

@monitor_app.get("/api/metrics/training")
async def get_training_metrics(days: int = Query(7, ge=1, le=90)):
    """Get training metrics"""
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'chatbot_training'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    collector = MetricsCollector(db_config)
    metrics = await collector.collect_training_metrics(days)
    
    return metrics

@monitor_app.get("/api/metrics/search")
async def get_search_metrics(days: int = Query(7, ge=1, le=90)):
    """Get search metrics"""
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'chatbot_training'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    collector = MetricsCollector(db_config)
    metrics = await collector.collect_search_metrics(days)
    
    return metrics

@monitor_app.get("/api/alerts")
async def get_active_alerts(chatbot_id: Optional[str] = Query(None)):
    """Get active alerts"""
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'chatbot_training'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    conn = await asyncpg.connect(**db_config)
    try:
        query = """
            SELECT * FROM system_alerts
            WHERE resolved_at IS NULL
            AND ($1 IS NULL OR chatbot_id = $1)
            ORDER BY created_at DESC
            LIMIT 50
        """
        
        alerts = await conn.fetch(query, chatbot_id)
        return [dict(alert) for alert in alerts]
        
    finally:
        await conn.close()

@monitor_app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    return prometheus_client.generate_latest()

# Start Prometheus metrics server
if __name__ == "__main__":
    # Start Prometheus metrics server on port 8000
    start_http_server(8000)
    
    # Start main API server
    import uvicorn
    uvicorn.run(monitor_app, host="0.0.0.0", port=8005)