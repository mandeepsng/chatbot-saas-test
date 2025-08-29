#!/bin/bash

# ChatBot SaaS Database Viewer
# Easy way to view database entries

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🗄️ ChatBot SaaS Database Viewer${NC}"
echo "=================================="

# Find PostgreSQL container
POSTGRES_CONTAINER=$(docker ps --format "table {{.Names}}" | grep postgres | head -1)

if [ -z "$POSTGRES_CONTAINER" ]; then
    echo "❌ PostgreSQL container not found!"
    echo "Make sure your services are running with: ./fix-deployment.sh"
    exit 1
fi

echo -e "${GREEN}📊 Database Overview${NC}"
echo "Container: $POSTGRES_CONTAINER"
echo "Database: chatbot_saas_db"
echo "User: chatbot_saas_user"
echo ""

# Function to run SQL query
run_query() {
    docker exec -it $POSTGRES_CONTAINER psql -U chatbot_saas_user -d chatbot_saas_db -c "$1"
}

# Show menu
show_menu() {
    echo -e "${YELLOW}Choose what to view:${NC}"
    echo "1. All Users"
    echo "2. Recent Signups (Last 10)"
    echo "3. User Statistics" 
    echo "4. All Tables"
    echo "5. Custom SQL Query"
    echo "6. Database Connection Test"
    echo "0. Exit"
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice (0-6): " choice
    echo ""
    
    case $choice in
        1)
            echo -e "${GREEN}👥 All Users:${NC}"
            run_query "SELECT id, email, company, plan_type, created_at::date as signup_date, last_login::date as last_login_date FROM users ORDER BY created_at DESC;"
            ;;
        2)
            echo -e "${GREEN}🔥 Recent Signups (Last 10):${NC}"
            run_query "SELECT email, company, created_at FROM users ORDER BY created_at DESC LIMIT 10;"
            ;;
        3)
            echo -e "${GREEN}📈 User Statistics:${NC}"
            run_query "SELECT COUNT(*) as total_users, COUNT(CASE WHEN created_at >= CURRENT_DATE - INTERVAL '7 days' THEN 1 END) as new_this_week, COUNT(CASE WHEN last_login >= CURRENT_DATE - INTERVAL '7 days' THEN 1 END) as active_this_week FROM users;"
            ;;
        4)
            echo -e "${GREEN}🗂️ All Tables:${NC}"
            run_query "\dt"
            ;;
        5)
            echo -e "${YELLOW}Enter your SQL query:${NC}"
            read -p "SQL> " custom_query
            if [ ! -z "$custom_query" ]; then
                run_query "$custom_query"
            fi
            ;;
        6)
            echo -e "${GREEN}🔌 Testing Database Connection:${NC}"
            run_query "SELECT 'Connection successful!' as status, NOW() as current_time;"
            ;;
        0)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid choice. Please try again."
            ;;
    esac
    echo ""
    read -p "Press Enter to continue..."
    echo ""
done