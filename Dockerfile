# Multi-stage Dockerfile for ChatFlow AI Backend
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY database/ ./database/

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/uploads && \
    chown -R app:app /app

# Switch to app user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Production stage
FROM base as production

# Copy startup script
COPY scripts/start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8001 8002 8003 8004 8005 8000

CMD ["/start.sh"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest==7.4.3 \
    pytest-asyncio==0.23.2 \
    pytest-mock==3.12.0 \
    black==23.12.1 \
    isort==5.13.2 \
    flake8==6.1.0 \
    mypy==1.8.0 \
    jupyter==1.0.0

# Enable development mode
ENV DEVELOPMENT=true

CMD ["uvicorn", "backend.data_ingestion:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

# Worker stage (for Celery workers)
FROM base as worker

CMD ["celery", "-A", "backend.embedding_service", "worker", "--loglevel=info", "--concurrency=4"]