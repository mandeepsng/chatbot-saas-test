#!/bin/bash

# ChatBot SaaS - DigitalOcean 4GB Deployment Script
# Run this script on your DigitalOcean droplet

set -e

echo "🚀 Starting ChatBot SaaS deployment on DigitalOcean 4GB droplet..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root"
   exit 1
fi

# Step 1: System Setup
print_status "Setting up system..."

# Add swap if not exists
if [[ ! -f /swapfile ]]; then
    print_status "Creating 2GB swap file..."
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' | tee -a /etc/fstab
    print_success "Swap file created"
else
    print_warning "Swap file already exists"
fi

# Update system
print_status "Updating system packages..."
apt update && apt upgrade -y
apt install -y curl wget git nginx certbot python3-certbot-nginx htop ufw

# Step 2: Install Docker
if ! command -v docker &> /dev/null; then
    print_status "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    print_success "Docker installed"
else
    print_warning "Docker already installed"
fi

# Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_status "Installing Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    print_success "Docker Compose installed"
else
    print_warning "Docker Compose already installed"
fi

# Step 3: Configure Firewall
print_status "Configuring firewall..."
ufw allow ssh
ufw allow 80
ufw allow 443
ufw --force enable
print_success "Firewall configured"

# Step 4: Create application directory
APP_DIR="/opt/chatbot-saas"
if [[ ! -d $APP_DIR ]]; then
    mkdir -p $APP_DIR
    print_success "Application directory created at $APP_DIR"
fi

cd $APP_DIR

# Step 5: Setup environment file
if [[ ! -f .env ]]; then
    print_status "Creating environment file..."
    
    # Generate secure passwords
    POSTGRES_PASSWORD=$(openssl rand -base64 32)
    JWT_SECRET=$(openssl rand -base64 32)
    
    cat > .env << EOF
# Database Configuration (separate from existing)
POSTGRES_DB=chatbot_saas_db
POSTGRES_USER=chatbot_saas_user
POSTGRES_PASSWORD=$POSTGRES_PASSWORD

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# JWT Configuration
JWT_SECRET_KEY=$JWT_SECRET
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440

# Environment
ENVIRONMENT=production
DEBUG=False

# OpenAI API (optional - leave empty to use local models)
OPENAI_API_KEY=

# Resource Limits (4GB optimized)
POSTGRES_MAX_CONNECTIONS=50
REDIS_MAXMEMORY=200mb
WORKER_CONCURRENCY=2

# Domain (replace with your domain)
DOMAIN=localhost
EOF

    print_success "Environment file created with secure passwords"
    print_warning "Please edit .env file and update DOMAIN with your actual domain"
else
    print_warning "Environment file already exists"
fi

# Step 6: Build and start services
print_status "Building and starting services..."

# Use the 4GB optimized compose file
docker-compose -f docker-compose.4gb.yml build
docker-compose -f docker-compose.4gb.yml up -d

print_success "Services started"

# Step 7: Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 30

# Check service health
print_status "Checking service health..."
for i in {1..30}; do
    if curl -f http://localhost:8091/health &>/dev/null; then
        print_success "Data Ingestion service is healthy"
        break
    fi
    sleep 2
done

for i in {1..30}; do
    if curl -f http://localhost:8092/health &>/dev/null; then
        print_success "Vector Search service is healthy"
        break
    fi
    sleep 2
done

for i in {1..30}; do
    if curl -f http://localhost:8093/health &>/dev/null; then
        print_success "Training Pipeline service is healthy"
        break
    fi
    sleep 2
done

# Step 8: Configure Nginx
print_status "Configuring Nginx..."

# Backup default nginx config
if [[ -f /etc/nginx/nginx.conf ]]; then
    cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup
fi

# Copy our optimized nginx config
cp nginx.4gb.conf /etc/nginx/nginx.conf

# Test nginx configuration
nginx -t

# Start nginx
systemctl enable nginx
systemctl restart nginx

print_success "Nginx configured and started"

# Step 9: Setup systemd service for auto-start
print_status "Setting up auto-start service..."

cat > /etc/systemd/system/chatbot-saas.service << EOF
[Unit]
Description=ChatBot SaaS Application
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$APP_DIR
ExecStart=docker-compose -f docker-compose.4gb.yml up -d
ExecStop=docker-compose -f docker-compose.4gb.yml down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable chatbot-saas.service

print_success "Auto-start service configured"

# Step 10: Setup monitoring
print_status "Setting up monitoring..."

mkdir -p /opt/scripts

cat > /opt/scripts/monitor.sh << 'EOF'
#!/bin/bash
echo "=== $(date) ==="
echo "=== System Resources ==="
free -h
echo "=== Docker Stats ==="
docker stats --no-stream
echo "=== Disk Usage ==="
df -h
echo "=== Service Status ==="
docker-compose -f /opt/chatbot-saas/docker-compose.4gb.yml ps
echo "================================"
EOF

chmod +x /opt/scripts/monitor.sh

# Add to crontab for regular monitoring
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/scripts/monitor.sh >> /var/log/resource-monitor.log") | crontab -

print_success "Monitoring setup complete"

# Step 11: Setup log rotation
print_status "Setting up log rotation..."

cat > /etc/logrotate.d/chatbot-saas << EOF
/var/lib/docker/containers/*/*.log {
    rotate 7
    daily
    compress
    size 50M
    missingok
    delaycompress
    copytruncate
}

/var/log/resource-monitor.log {
    rotate 7
    daily
    compress
    size 10M
    missingok
    delaycompress
    copytruncate
}
EOF

print_success "Log rotation configured"

# Step 12: Final status check
print_status "Performing final status check..."

sleep 10

echo ""
echo "=== SERVICE STATUS ==="
docker-compose -f docker-compose.4gb.yml ps

echo ""
echo "=== SYSTEM RESOURCES ==="
free -h

echo ""
echo "=== DISK USAGE ==="
df -h

echo ""
print_success "🎉 Deployment completed successfully!"
echo ""
print_status "Access your ChatBot SaaS application at:"
echo "  • Frontend: http://$(curl -s ifconfig.me):8090"
echo "  • Dashboard: http://$(curl -s ifconfig.me):8090/dashboard.html"
echo "  • Auth: http://$(curl -s ifconfig.me):8090/auth.html"
echo ""
print_status "API Endpoints:"
echo "  • Data Ingestion: http://$(curl -s ifconfig.me):8090/api/health"
echo "  • Vector Search: http://$(curl -s ifconfig.me):8090/search/health" 
echo "  • Training Pipeline: http://$(curl -s ifconfig.me):8090/train/health"
echo ""
print_status "Your existing services remain unaffected:"
echo "  • pradmin4: continues on port 8080"
echo "  • postgres_pgvector: continues on port 5432"
echo "  • embedding-service: continues on port 8000"
echo ""
print_warning "Next steps:"
echo "  1. Update DOMAIN in .env file with your actual domain"
echo "  2. Setup SSL with: certbot --nginx -d your-domain.com"
echo "  3. Monitor logs with: docker-compose -f docker-compose.4gb.yml logs -f"
echo "  4. Monitor resources with: /opt/scripts/monitor.sh"
echo ""
print_status "For troubleshooting, check /var/log/resource-monitor.log"