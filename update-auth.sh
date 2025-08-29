#!/bin/bash

echo "🔄 Updating ChatBot SaaS with authentication..."

# Stop services 
docker-compose -f docker-compose.minimal.yml down

# Rebuild data-ingestion service with auth endpoints
echo "🏗️ Rebuilding services with auth endpoints..."
docker-compose -f docker-compose.minimal.yml build data-ingestion

# Start services
echo "🚀 Starting updated services..."
docker-compose -f docker-compose.minimal.yml up -d

echo "⏳ Waiting for services to be ready..."
sleep 30

# Test health endpoints
echo "🧪 Testing services..."

# Test data ingestion health
if curl -f http://localhost:8091/health 2>/dev/null; then
    echo "✅ Data Ingestion API: Healthy"
else
    echo "❌ Data Ingestion API: Not ready"
fi

# Test auth endpoints through nginx
if curl -f http://localhost:8090/api/health 2>/dev/null; then
    echo "✅ Auth API via Nginx: Accessible"
else
    echo "❌ Auth API via Nginx: Issues"
fi

# Test signup endpoint
echo "🧪 Testing signup endpoint..."
response=$(curl -s -X POST http://localhost:8090/api/auth/signup \
    -H "Content-Type: application/json" \
    -d '{"email":"test@example.com","password":"test123","company":"Test Company"}' \
    -w "%{http_code}")

if [[ $response == *"200"* ]] || [[ $response == *"400"* ]]; then
    echo "✅ Signup endpoint: Responding"
else
    echo "❌ Signup endpoint: Not working (Response: $response)"
fi

echo ""
echo "📊 Container Status:"
docker-compose -f docker-compose.minimal.yml ps

echo ""
echo "🌐 Access your updated ChatBot SaaS:"
echo "  • Main site: http://206.189.131.217:8090"
echo "  • Sign Up: http://206.189.131.217:8090/auth/signup.html"
echo "  • Sign In: http://206.189.131.217:8090/auth/signin.html"

echo ""
echo "🔧 If auth still doesn't work, check logs with:"
echo "  docker-compose -f docker-compose.minimal.yml logs data-ingestion"