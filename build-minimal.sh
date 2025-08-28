#!/bin/bash

# Minimal build script for 4GB droplet - no dependency conflicts
set -e

echo "🔧 Building minimal ChatBot SaaS for 4GB droplet..."

# Clean up
docker system prune -f
docker rmi $(docker images -q "*chatbot*") 2>/dev/null || true

# Build minimal version
echo "🏗️ Building minimal services..."
docker-compose -f docker-compose.minimal.yml build --no-cache

if [ $? -eq 0 ]; then
    echo "✅ Build successful! Starting minimal services..."
    docker-compose -f docker-compose.minimal.yml up -d
    
    echo "⏳ Waiting for services..."
    sleep 45
    
    echo "🧪 Testing services..."
    
    # Test data ingestion
    for i in {1..12}; do
        if curl -f http://localhost:8091/health 2>/dev/null; then
            echo "✅ Data Ingestion (port 8091) - Healthy"
            break
        fi
        echo "⏳ Waiting for Data Ingestion... ($i/12)"
        sleep 5
    done
    
    # Test vector search
    for i in {1..12}; do
        if curl -f http://localhost:8092/health 2>/dev/null; then
            echo "✅ Vector Search (port 8092) - Healthy"
            break
        fi
        echo "⏳ Waiting for Vector Search... ($i/12)"
        sleep 5
    done
    
    # Test training pipeline
    for i in {1..12}; do
        if curl -f http://localhost:8093/health 2>/dev/null; then
            echo "✅ Training Pipeline (port 8093) - Healthy"
            break
        fi
        echo "⏳ Waiting for Training Pipeline... ($i/12)"
        sleep 5
    done
    
    # Test frontend
    if curl -f http://localhost:8090 2>/dev/null; then
        echo "✅ Frontend (port 8090) - Accessible"
    else
        echo "⚠️ Frontend may still be starting"
    fi
    
    echo ""
    echo "📊 Service Status:"
    docker-compose -f docker-compose.minimal.yml ps
    
    echo ""
    echo "🎉 ChatBot SaaS Minimal Deployment Ready!"
    echo ""
    echo "🌐 Access your application:"
    echo "  • Frontend: http://localhost:8090"
    echo "  • Dashboard: http://localhost:8090/dashboard.html"
    echo "  • Admin: http://localhost:8090/admin/dashboard.html"
    echo ""
    echo "📡 API Endpoints:"
    echo "  • Data Ingestion: http://localhost:8090/api/"
    echo "  • Vector Search: http://localhost:8090/search/"
    echo "  • Training: http://localhost:8090/train/"
    echo ""
    echo "💾 Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
    
else
    echo "❌ Build failed!"
    exit 1
fi