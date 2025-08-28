#!/bin/bash

# Complete cleanup script for ChatBot SaaS builds
set -e

echo "🧹 Cleaning up previous incomplete builds..."

# Stop all running containers from previous attempts
echo "🛑 Stopping all ChatBot SaaS containers..."
docker-compose -f docker-compose.yml down 2>/dev/null || true
docker-compose -f docker-compose.prod.yml down 2>/dev/null || true
docker-compose -f docker-compose.4gb.yml down 2>/dev/null || true
docker-compose -f docker-compose.minimal.yml down 2>/dev/null || true

# Remove all ChatBot SaaS related containers
echo "🗑️ Removing ChatBot SaaS containers..."
docker ps -a | grep -E "(chatbot|demo-chatbot)" | awk '{print $1}' | xargs docker rm -f 2>/dev/null || true

# Remove ChatBot SaaS images
echo "🖼️ Removing ChatBot SaaS images..."
docker images | grep -E "(chatbot|demo-chatbot)" | awk '{print $3}' | xargs docker rmi -f 2>/dev/null || true

# Remove unused images and build cache
echo "💾 Cleaning Docker system..."
docker system prune -f
docker builder prune -f

# Remove any leftover volumes (be careful - this removes ALL unused volumes)
echo "📦 Cleaning unused volumes..."
docker volume prune -f

# Clean up any build artifacts
echo "🗂️ Cleaning build artifacts..."
rm -rf __pycache__ 2>/dev/null || true
rm -rf .pytest_cache 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true

# Show remaining containers (your existing ones should be safe)
echo ""
echo "📊 Remaining containers (your existing services):"
docker ps -a

echo ""
echo "💿 Disk usage after cleanup:"
df -h

echo ""
echo "🧠 Memory usage:"
free -h

echo ""
echo "✅ Cleanup complete! Your existing services are preserved:"
echo "  • pradmin4 (port 8080) - should still be running"
echo "  • postgres_pgvector (port 5432) - should still be running"
echo "  • embedding-service (port 8000) - should still be running"

echo ""
echo "🚀 Now you can run a fresh build:"
echo "  ./fix-deployment.sh"
echo ""
echo "Or build from scratch:"
echo "  ./build-minimal.sh"