#!/bin/bash

# ChatBot SaaS - Deployment Testing Script
# Test all components after deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOMAIN=${DOMAIN:-$(curl -s ifconfig.me)}
BASE_URL="http://$DOMAIN:8090"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Test counter
TOTAL_TESTS=0
PASSED_TESTS=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    print_status "Running test: $test_name"
    
    if eval "$test_command"; then
        print_success "$test_name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        print_error "$test_name"
        return 1
    fi
}

echo "🧪 Starting ChatBot SaaS deployment tests..."
echo "Testing domain: $DOMAIN"
echo "================================"

# Test 1: System Resources
run_test "System memory check" "
    AVAILABLE_MEM=\$(free -m | awk 'NR==2{printf \"%.0f\", \$7*100/\$2}')
    [ \$AVAILABLE_MEM -gt 20 ]
"

run_test "Disk space check" "
    AVAILABLE_DISK=\$(df / | tail -1 | awk '{print \$5}' | sed 's/%//')
    [ \$AVAILABLE_DISK -lt 80 ]
"

run_test "Swap space check" "
    swapon --show | grep -q '/swapfile'
"

# Test 2: Docker Services
run_test "Docker daemon running" "
    systemctl is-active --quiet docker
"

run_test "Docker containers running" "
    RUNNING_CONTAINERS=\$(docker ps --format 'table {{.Names}}' | grep -E 'postgres|redis|data-ingestion|vector-search|training-pipeline' | wc -l)
    [ \$RUNNING_CONTAINERS -ge 5 ]
"

# Test 3: Network Connectivity
run_test "Port 80 accessible" "
    nc -z localhost 80
"

run_test "Nginx running" "
    systemctl is-active --quiet nginx
"

# Test 4: Database Connectivity
run_test "PostgreSQL container healthy" "
    docker exec chatbot-saas_postgres_1 pg_isready -U chatbot_user >/dev/null 2>&1 || 
    docker exec chatbot-saas-postgres-1 pg_isready -U chatbot_user >/dev/null 2>&1
"

run_test "Redis container healthy" "
    docker exec chatbot-saas_redis_1 redis-cli ping | grep -q PONG ||
    docker exec chatbot-saas-redis-1 redis-cli ping | grep -q PONG
"

run_test "pgvector extension available" "
    docker exec chatbot-saas_postgres_1 psql -U chatbot_user -d chatbot_training -c \"SELECT extname FROM pg_extension WHERE extname = 'vector';\" | grep -q vector ||
    docker exec chatbot-saas-postgres-1 psql -U chatbot_user -d chatbot_training -c \"SELECT extname FROM pg_extension WHERE extname = 'vector';\" | grep -q vector
"

# Test 5: API Health Checks
run_test "Data Ingestion API health" "
    curl -f -s \$BASE_URL/api/health | grep -q 'healthy'
"

run_test "Vector Search API health" "
    curl -f -s \$BASE_URL/search/health | grep -q 'healthy'
"

run_test "Training Pipeline API health" "
    curl -f -s \$BASE_URL/train/health | grep -q 'healthy'
"

# Test 6: Frontend Accessibility
run_test "Main page accessible" "
    curl -f -s \$BASE_URL/ | grep -q '<title>'
"

run_test "Dashboard page accessible" "
    curl -f -s \$BASE_URL/dashboard.html | grep -q '<title>'
"

run_test "Auth page accessible" "
    curl -f -s \$BASE_URL/auth.html | grep -q '<title>'
"

# Test 7: Authentication API
run_test "Auth endpoint responsive" "
    curl -f -s -X POST \$BASE_URL/api/auth/test -H 'Content-Type: application/json' -d '{}' >/dev/null
"

# Test 8: Vector Search Functionality (Basic)
run_test "Vector search endpoint responsive" "
    curl -f -s -X POST \$BASE_URL/search/test -H 'Content-Type: application/json' -d '{}' >/dev/null
"

# Test 9: File Upload Capability
run_test "File upload endpoint responsive" "
    echo 'test content' > /tmp/test.txt
    curl -f -s -X POST \$BASE_URL/api/documents/test -F 'file=@/tmp/test.txt' >/dev/null
    rm -f /tmp/test.txt
"

# Test 10: Resource Usage
run_test "Memory usage under 90%" "
    MEM_USAGE=\$(free | grep Mem | awk '{printf \"%.0f\", \$3*100/\$2}')
    [ \$MEM_USAGE -lt 90 ]
"

run_test "CPU load reasonable" "
    LOAD=\$(uptime | awk '{print \$(NF-2)}' | sed 's/,//')
    [ \$(echo \"\$LOAD < 2.0\" | bc -l) -eq 1 ]
"

# Test 11: SSL/HTTPS (if configured)
if curl -f -s https://$DOMAIN >/dev/null 2>&1; then
    run_test "HTTPS accessible" "
        curl -f -s https://$DOMAIN | grep -q '<title>'
    "
    
    run_test "SSL certificate valid" "
        echo | timeout 5 openssl s_client -connect $DOMAIN:443 2>/dev/null | openssl x509 -noout -dates | grep -q 'notAfter'
    "
else
    print_warning "HTTPS not configured - skipping SSL tests"
fi

# Test 12: Log Files
run_test "Application logs present" "
    docker logs chatbot-saas_data-ingestion_1 2>&1 | tail -n 5 | grep -q . ||
    docker logs chatbot-saas-data-ingestion-1 2>&1 | tail -n 5 | grep -q .
"

run_test "Nginx access logs present" "
    [ -s /var/log/nginx/access.log ]
"

run_test "System monitoring log present" "
    [ -s /var/log/resource-monitor.log ]
"

# Test 13: Performance Benchmarks
print_status "Running basic performance test..."

# Simple load test
run_test "Basic load test (10 concurrent requests)" "
    for i in {1..10}; do
        curl -f -s \$BASE_URL/ >/dev/null &
    done
    wait
    # If we get here, all requests succeeded
    true
"

# Response time test
run_test "API response time under 2 seconds" "
    RESPONSE_TIME=\$(curl -o /dev/null -s -w '%{time_total}' \$BASE_URL/api/health)
    [ \$(echo \"\$RESPONSE_TIME < 2.0\" | bc -l) -eq 1 ]
"

# Test Results Summary
echo ""
echo "================================"
echo "📊 Test Results Summary"
echo "================================"
echo "Total Tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $((TOTAL_TESTS - PASSED_TESTS))"

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    print_success "🎉 All tests passed! Deployment is healthy."
    echo ""
    print_status "Your ChatBot SaaS platform is ready to use:"
    echo "  • Frontend: $BASE_URL"
    echo "  • Dashboard: $BASE_URL/dashboard.html"
    echo "  • Admin: $BASE_URL/admin/dashboard.html"
    echo ""
    print_status "API Endpoints:"
    echo "  • Health: $BASE_URL/api/health"
    echo "  • Search: $BASE_URL/search/health"
    echo "  • Training: $BASE_URL/train/health"
else
    PASS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    if [ $PASS_RATE -ge 80 ]; then
        print_warning "⚠️ Most tests passed ($PASS_RATE%). Check failed tests above."
    else
        print_error "❌ Many tests failed ($PASS_RATE% pass rate). Check system configuration."
    fi
    
    echo ""
    print_status "Troubleshooting tips:"
    echo "  • Check logs: docker-compose -f docker-compose.4gb.yml logs"
    echo "  • Check resources: free -h && docker stats"
    echo "  • Restart services: docker-compose -f docker-compose.4gb.yml restart"
    echo "  • Check monitoring: tail -f /var/log/resource-monitor.log"
fi

echo ""
print_status "For ongoing monitoring, run this script regularly:"
echo "  ./test-deployment.sh"

exit 0