#!/bin/bash
set -e

echo "Starting ChatFlow AI Backend Services..."

# Wait for database to be ready
echo "Waiting for database..."
until pg_isready -h "${DB_HOST:-postgres}" -p "${DB_PORT:-5432}" -U "${DB_USER:-postgres}"; do
  echo "Database is unavailable - sleeping"
  sleep 1
done
echo "Database is ready!"

# Wait for Redis to be ready
echo "Waiting for Redis..."
until redis-cli -h "${REDIS_HOST:-redis}" -p "${REDIS_PORT:-6379}" ping; do
  echo "Redis is unavailable - sleeping"
  sleep 1
done
echo "Redis is ready!"

# Run database migrations if needed
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    # Add migration command here if using Alembic
    # alembic upgrade head
fi

# Start the appropriate service based on SERVICE_NAME
case "${SERVICE_NAME}" in
    "data-ingestion")
        echo "Starting Data Ingestion Service..."
        exec uvicorn backend.data_ingestion:app --host 0.0.0.0 --port 8001
        ;;
    "vector-search")
        echo "Starting Vector Search Service..."
        exec uvicorn backend.vector_search:search_app --host 0.0.0.0 --port 8002
        ;;
    "training-pipeline")
        echo "Starting Training Pipeline Service..."
        exec uvicorn backend.training_pipeline:training_app --host 0.0.0.0 --port 8003
        ;;
    "performance-optimizer")
        echo "Starting Performance Optimizer Service..."
        exec uvicorn backend.performance_optimizer:perf_app --host 0.0.0.0 --port 8004
        ;;
    "monitoring-analytics")
        echo "Starting Monitoring & Analytics Service..."
        exec python backend/monitoring_analytics.py
        ;;
    "celery-worker-embeddings")
        echo "Starting Celery Worker for Embeddings..."
        exec celery -A backend.embedding_service worker --loglevel=info --concurrency=4 -Q embeddings
        ;;
    "celery-worker-training")
        echo "Starting Celery Worker for Training..."
        exec celery -A backend.training_pipeline worker --loglevel=info --concurrency=2 -Q training
        ;;
    "celery-worker-optimization")
        echo "Starting Celery Worker for Optimization..."
        exec celery -A backend.performance_optimizer worker --loglevel=info --concurrency=2 -Q optimization
        ;;
    *)
        echo "Starting default service (Data Ingestion)..."
        exec uvicorn backend.data_ingestion:app --host 0.0.0.0 --port 8001
        ;;
esac