#!/bin/bash

# Quick Database Commands for ChatBot SaaS

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Find PostgreSQL container
POSTGRES_CONTAINER=$(docker ps --format "table {{.Names}}" | grep postgres | head -1)

if [ -z "$POSTGRES_CONTAINER" ]; then
    echo "❌ PostgreSQL container not found!"
    exit 1
fi

echo -e "${BLUE}Quick Database Commands${NC}"
echo "======================="

# Function to run SQL
run_sql() {
    docker exec $POSTGRES_CONTAINER psql -U chatbot_saas_user -d chatbot_saas_db -c "$1"
}

# Quick commands
case "$1" in
    "users")
        echo -e "${GREEN}👥 All Users:${NC}"
        run_sql "SELECT email, company, created_at FROM users ORDER BY created_at DESC;"
        ;;
    "count")
        echo -e "${GREEN}📊 User Count:${NC}"
        run_sql "SELECT COUNT(*) as total_users FROM users;"
        ;;
    "recent")
        echo -e "${GREEN}🔥 Recent Users:${NC}"
        run_sql "SELECT email, company, created_at FROM users WHERE created_at >= CURRENT_DATE - INTERVAL '24 hours' ORDER BY created_at DESC;"
        ;;
    "tables")
        echo -e "${GREEN}🗂️ All Tables:${NC}"
        run_sql "\dt"
        ;;
    "schema")
        echo -e "${GREEN}🏗️ Users Table Schema:${NC}"
        run_sql "\d users"
        ;;
    *)
        echo "Usage: $0 [command]"
        echo ""
        echo "Available commands:"
        echo "  users   - Show all users"
        echo "  count   - Show user count"
        echo "  recent  - Show users from last 24 hours"
        echo "  tables  - Show all database tables"
        echo "  schema  - Show users table structure"
        echo ""
        echo "Examples:"
        echo "  ./quick-db-commands.sh users"
        echo "  ./quick-db-commands.sh count"
        ;;
esac