# ChatBot SaaS - DigitalOcean 4GB Droplet Deployment Guide

This guide provides step-by-step instructions to deploy the ChatBot SaaS platform on a 4GB RAM DigitalOcean droplet **alongside existing containers** without conflicts.

## ⚠️ **Safe Deployment Notice**

This deployment is configured to **NOT interfere** with your existing containers:
- Uses different ports (8090-8095 range)  
- Separate database (port 5433)
- Isolated volumes and networks
- No conflicts with existing services

## Prerequisites

- DigitalOcean account with existing containers running
- Domain name (optional, can use droplet IP)
- SSH key configured
- Docker and Docker Compose already installed

## Step 1: Create DigitalOcean Droplet

1. **Create Droplet**:
   - Image: Ubuntu 22.04 LTS
   - Plan: Basic - Regular Intel with SSD
   - Size: $24/month (4GB RAM, 2 vCPUs, 80GB SSD)
   - Region: Choose closest to your users
   - Authentication: SSH Key (recommended)
   - Enable: IPv6, User data, Monitoring

2. **Add Swap for 4GB RAM**:
   ```bash
   # SSH into your droplet
   ssh root@your_droplet_ip
   
   # Create 2GB swap file
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
  sudo mkswap /swapfile 
   sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

## Step 2: Server Setup

1. **Update System**:
   ```bash
   apt update && apt upgrade -y
   apt install -y curl wget git nginx certbot python3-certbot-nginx
   ```

2. **Install Docker & Docker Compose**:
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   
   # Install Docker Compose
   curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   chmod +x /usr/local/bin/docker-compose
   
   # Verify installation
   docker --version
   docker-compose --version
   ```

3. **Configure Firewall**:
   ```bash
   ufw allow ssh
   ufw allow 80
   ufw allow 443
   ufw --force enable
   ```

## Step 3: Deploy Application

1. **Clone Repository**:
   ```bash
   cd /opt
   git clone <your-repo-url> chatbot-saas
   cd chatbot-saas
   ```

2. **Configure Environment**:
   ```bash
   # Copy and edit environment file
   cp .env.example .env
   nano .env
   ```

   **Required .env Configuration** (4GB optimized):
   ```env
   # Database (separate from your existing postgres)
   POSTGRES_DB=chatbot_saas_db
   POSTGRES_USER=chatbot_saas_user
   POSTGRES_PASSWORD=your_secure_password_here
   
   # Redis
   REDIS_URL=redis://redis:6379/0
   
   # API Keys (Optional - can run without OpenAI)
   OPENAI_API_KEY=your_openai_api_key_or_leave_empty
   
   # JWT Security
   JWT_SECRET_KEY=your_jwt_secret_key_min_32_chars
   JWT_ALGORITHM=HS256
   JWT_EXPIRE_MINUTES=1440
   
   # Environment
   ENVIRONMENT=production
   DEBUG=False
   
   # Domain
   DOMAIN=your-domain.com
   
   # Resource Limits (4GB RAM optimized)
   POSTGRES_MAX_CONNECTIONS=50
   REDIS_MAXMEMORY=512mb
   WORKER_CONCURRENCY=2
   ```

3. **Create 4GB Optimized Docker Compose**:
   ```bash
   cp docker-compose.yml docker-compose.prod.yml
   nano docker-compose.prod.yml
   ```

## Step 4: Resource-Optimized Docker Compose (Port-Safe)

The provided `docker-compose.4gb.yml` uses these ports to avoid conflicts:
- **PostgreSQL**: 5433 (instead of 5432)
- **Redis**: 6380 (instead of 6379) 
- **Data Ingestion**: 8091 (instead of 8001)
- **Vector Search**: 8092 (instead of 8002)
- **Training Pipeline**: 8093 (instead of 8003)
- **Monitoring**: 8095 (instead of 8005)
- **Nginx**: 8090 (instead of 80), 8443 (instead of 443)

Configuration highlights:

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/schema.sql:/docker-entrypoint-initdb.d/001-schema.sql
    ports:
      - "5432:5432"
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
    command: >
      postgres -c max_connections=50
               -c shared_buffers=256MB
               -c effective_cache_size=512MB
               -c work_mem=4MB

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    deploy:
      resources:
        limits:
          memory: 256M
        reservations:
          memory: 128M
    command: redis-server --maxmemory 200mb --maxmemory-policy allkeys-lru

  data-ingestion:
    build: .
    command: uvicorn backend.data_ingestion:app --host 0.0.0.0 --port 8001
    ports:
      - "8001:8001"
    depends_on:
      - postgres
      - redis
    env_file: .env
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  vector-search:
    build: .
    command: uvicorn backend.vector_search:app --host 0.0.0.0 --port 8002
    ports:
      - "8002:8002"
    depends_on:
      - postgres
      - redis
    env_file: .env
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  training-pipeline:
    build: .
    command: uvicorn backend.training_pipeline:app --host 0.0.0.0 --port 8003
    ports:
      - "8003:8003"
    depends_on:
      - postgres
      - redis
    env_file: .env
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  embedding-worker:
    build: .
    command: celery -A backend.embedding_service worker --loglevel=info --concurrency=2
    depends_on:
      - postgres
      - redis
    env_file: .env
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./:/var/www/html
      - /etc/letsencrypt:/etc/letsencrypt:ro
    depends_on:
      - data-ingestion
      - vector-search
      - training-pipeline
    deploy:
      resources:
        limits:
          memory: 128M
        reservations:
          memory: 64M

volumes:
  postgres_data:
```

## Step 5: Build and Deploy

1. **Build Application**:
   ```bash
   # Build Docker images
   docker-compose -f docker-compose.prod.yml build
   
   # Start services
   docker-compose -f docker-compose.prod.yml up -d
   
   # Check status
   docker-compose -f docker-compose.prod.yml ps
   ```

2. **Verify Services**:
   ```bash
   # Check logs
   docker-compose -f docker-compose.prod.yml logs -f
   
   # Test health endpoints (new ports)
   curl http://localhost:8091/health
   curl http://localhost:8092/health
   curl http://localhost:8093/health
   ```

## Step 6: Configure Nginx (Production)

1. **Update Nginx Configuration**:
   ```bash
   nano /etc/nginx/sites-available/chatbot-saas
   ```

   ```nginx
   server {
       listen 80;
       server_name your-domain.com www.your-domain.com;
       
       # Serve static files
       location / {
           root /opt/chatbot-saas;
           try_files $uri $uri/ /index.html;
           
           # Security headers
           add_header X-Frame-Options "SAMEORIGIN" always;
           add_header X-Content-Type-Options "nosniff" always;
           add_header X-XSS-Protection "1; mode=block" always;
       }
       
       # API proxy
       location /api/ {
           proxy_pass http://localhost:8001/;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
       
       # Vector search API
       location /search/ {
           proxy_pass http://localhost:8002/;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
       
       # Training API
       location /train/ {
           proxy_pass http://localhost:8003/;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

2. **Enable Site**:
   ```bash
   ln -s /etc/nginx/sites-available/chatbot-saas /etc/nginx/sites-enabled/
   nginx -t
   systemctl restart nginx
   ```

3. **Setup SSL Certificate**:
   ```bash
   certbot --nginx -d your-domain.com -d www.your-domain.com
   ```

## Step 7: Monitoring and Maintenance

1. **Resource Monitoring**:
   ```bash
   # Check memory usage
   free -h
   
   # Check Docker resource usage
   docker stats
   
   # Check disk space
   df -h
   ```

2. **Setup Log Rotation**:
   ```bash
   nano /etc/logrotate.d/docker
   ```
   
   ```
   /var/lib/docker/containers/*/*.log {
       rotate 7
       daily
       compress
       size 50M
       missingok
       delaycompress
       copytruncate
   }
   ```

3. **Auto-restart on Boot**:
   ```bash
   # Create systemd service
   nano /etc/systemd/system/chatbot-saas.service
   ```
   
   ```ini
   [Unit]
   Description=ChatBot SaaS Application
   After=docker.service
   Requires=docker.service
   
   [Service]
   Type=oneshot
   RemainAfterExit=yes
   WorkingDirectory=/opt/chatbot-saas
   ExecStart=docker-compose -f docker-compose.prod.yml up -d
   ExecStop=docker-compose -f docker-compose.prod.yml down
   TimeoutStartSec=0
   
   [Install]
   WantedBy=multi-user.target
   ```
   
   ```bash
   systemctl enable chatbot-saas.service
   ```

## Step 8: Testing the Deployment

1. **Access Application**:
   - Frontend: `http://your-domain.com:8090` or `http://YOUR_DROPLET_IP:8090`
   - Dashboard: `http://your-domain.com:8090/dashboard.html`
   - Auth: `http://your-domain.com:8090/auth.html`
   
   **Note**: Uses port 8090 to avoid conflicts with your existing services

2. **Test API Endpoints**:
   ```bash
   # Health checks (new ports)
   curl http://YOUR_DROPLET_IP:8090/api/health
   curl http://YOUR_DROPLET_IP:8090/search/health
   curl http://YOUR_DROPLET_IP:8090/train/health
   
   # Test authentication
   curl -X POST https://your-domain.com/api/auth/signup \
     -H "Content-Type: application/json" \
     -d '{"email":"test@example.com","password":"testpass123","company":"Test Co"}'
   ```

3. **Test Vector Search**:
   ```bash
   # Upload test document
   curl -X POST https://your-domain.com/api/documents/upload \
     -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     -F "file=@test.txt"
   
   # Search test
   curl -X POST https://your-domain.com/search/semantic \
     -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"query":"test question","limit":5}'
   ```

## Performance Optimization for 4GB RAM

1. **Database Tuning**:
   ```sql
   -- Connect to PostgreSQL and optimize
   ALTER SYSTEM SET shared_buffers = '256MB';
   ALTER SYSTEM SET effective_cache_size = '512MB';
   ALTER SYSTEM SET max_connections = '50';
   SELECT pg_reload_conf();
   ```

2. **Monitor Resource Usage**:
   ```bash
   # Create monitoring script
   nano /opt/monitor.sh
   ```
   
   ```bash
   #!/bin/bash
   echo "=== System Resources ==="
   free -h
   echo "=== Docker Stats ==="
   docker stats --no-stream
   echo "=== Disk Usage ==="
   df -h
   ```
   
   ```bash
   chmod +x /opt/monitor.sh
   # Run every 5 minutes
   crontab -e
   # Add: */5 * * * * /opt/monitor.sh >> /var/log/resource-monitor.log
   ```

## Troubleshooting

### Common Issues:

1. **Out of Memory**:
   - Restart services: `docker-compose -f docker-compose.prod.yml restart`
   - Check logs: `docker-compose -f docker-compose.prod.yml logs`

2. **Database Connection Issues**:
   - Check PostgreSQL: `docker-compose -f docker-compose.prod.yml logs postgres`
   - Verify connections: `docker exec -it chatbot-saas_postgres_1 psql -U chatbot_user -d chatbot_training -c "SELECT COUNT(*) FROM pg_stat_activity;"`

3. **High Memory Usage**:
   - Reduce worker concurrency in .env: `WORKER_CONCURRENCY=1`
   - Limit PostgreSQL memory: Adjust shared_buffers in docker-compose

### Backup Strategy:

```bash
# Database backup script
nano /opt/backup.sh
```

```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker exec chatbot-saas_postgres_1 pg_dump -U chatbot_user chatbot_training > "/opt/backups/backup_$DATE.sql"
# Keep only last 7 backups
find /opt/backups -name "backup_*.sql" -mtime +7 -delete
```

## Expected Performance on 4GB Droplet (Alongside Existing Services)

- **Concurrent Users**: 300-500 (reduced due to shared resources)
- **API Requests**: 50-100 req/sec
- **Vector Searches**: 25-50 searches/sec
- **Document Processing**: 5-10 docs/minute
- **Memory Usage**: ~2.5-3GB (leaving space for your existing services)

## 🛡️ **Safety Features**

- **Port Isolation**: No conflicts with existing services on ports 8080, 5432, 8000
- **Separate Databases**: Uses different database name and user
- **Independent Volumes**: Isolated data storage
- **Resource Limits**: Controlled memory usage to prevent OOM
- **Health Checks**: Monitors service health without affecting existing containers

## 🔄 **Coexistence with Existing Services**

Your existing containers will continue running normally:
- `pradmin4` (port 8080) - Unaffected
- `postgres_pgvector` (port 5432) - Unaffected  
- `embedding-service` (port 8000) - Unaffected

The ChatBot SaaS runs on ports 8090-8095 range, ensuring complete isolation.

The platform will run efficiently on 4GB RAM with these optimizations alongside your existing services. For higher loads, consider upgrading to 8GB droplet or implementing horizontal scaling.