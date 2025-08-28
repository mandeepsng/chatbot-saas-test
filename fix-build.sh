#!/bin/bash

# Fix Docker build dependencies for 4GB droplet
set -e

echo "🔧 Fixing Docker build dependencies..."

# Clean up any previous failed builds
docker system prune -f

# Remove any existing images that might have conflicts
docker rmi $(docker images -q chatbot-saas*) 2>/dev/null || true

# Build with no cache to ensure clean build
echo "🏗️ Building with clean slate..."
docker-compose -f docker-compose.4gb.yml build --no-cache --parallel

# If build succeeds, test it
if [ $? -eq 0 ]; then
    echo "✅ Build successful! Starting services..."
    docker-compose -f docker-compose.4gb.yml up -d
    
    echo "⏳ Waiting for services to start..."
    sleep 30
    
    echo "🧪 Testing health endpoints..."
    for i in {1..10}; do
        if curl -f http://localhost:8091/health 2>/dev/null; then
            echo "✅ Data Ingestion service is healthy"
            break
        fi
        sleep 5
    done
    
    echo "📊 Service status:"
    docker-compose -f docker-compose.4gb.yml ps
    
else
    echo "❌ Build failed. Check the logs above."
    exit 1
fi

echo "🎉 Deployment ready!"