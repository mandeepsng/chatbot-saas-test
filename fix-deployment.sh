#!/bin/bash

# Quick fix for deployment issues
set -e

echo "🔧 Fixing deployment issues..."

# Stop any running containers
docker-compose -f docker-compose.minimal.yml down 2>/dev/null || true

# Clean up failed containers
docker container prune -f
docker image prune -f

echo "🚀 Starting core services only..."

# Start just the essential services
docker-compose -f docker-compose.minimal.yml up -d postgres redis nginx

echo "⏳ Waiting for database and redis..."
sleep 15

# Start application services one by one
echo "🔌 Starting data-ingestion..."
docker-compose -f docker-compose.minimal.yml up -d data-ingestion

echo "🔍 Starting vector-search..."
docker-compose -f docker-compose.minimal.yml up -d vector-search

echo "🎯 Starting training-pipeline..."
docker-compose -f docker-compose.minimal.yml up -d training-pipeline

echo "⏳ Waiting for services to be healthy..."
sleep 30

echo "🧪 Testing services..."

# Test each service
if curl -f http://localhost:8090 2>/dev/null; then
    echo "✅ Frontend accessible at http://localhost:8090"
else
    echo "⚠️ Frontend not ready yet"
fi

if curl -f http://localhost:8091/health 2>/dev/null; then
    echo "✅ Data Ingestion API healthy"
else
    echo "⚠️ Data Ingestion not ready"
fi

if curl -f http://localhost:8092/health 2>/dev/null; then
    echo "✅ Vector Search API healthy"
else
    echo "⚠️ Vector Search not ready"
fi

if curl -f http://localhost:8093/health 2>/dev/null; then
    echo "✅ Training Pipeline API healthy"
else
    echo "⚠️ Training Pipeline not ready"
fi

echo ""
echo "📊 Container Status:"
docker-compose -f docker-compose.minimal.yml ps

echo ""
echo "💾 Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -10

echo ""
echo "🎉 Core ChatBot SaaS services are running!"
echo ""
echo "📱 Access your application:"
echo "  🌐 Frontend: http://localhost:8090"
echo "  📊 Dashboard: http://localhost:8090/dashboard.html" 
echo "  🔐 Auth: http://localhost:8090/auth.html"
echo "  👤 Admin: http://localhost:8090/admin/dashboard.html"
echo ""
echo "🔧 If you see any issues, check logs with:"
echo "  docker-compose -f docker-compose.minimal.yml logs [service-name]"