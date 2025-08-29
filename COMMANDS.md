# ChatBot SaaS - Quick Command Reference

## 🚀 Essential Commands

### **Start/Stop Services**
```bash
# Start ChatBot SaaS
./fix-deployment.sh

# Stop ChatBot SaaS (safe - preserves existing services)
docker-compose -f docker-compose.minimal.yml down

# Full cleanup and restart
./cleanup.sh && ./fix-deployment.sh
```

### **Service Management**
```bash
# Check service status
docker-compose -f docker-compose.minimal.yml ps

# Restart specific service
docker-compose -f docker-compose.minimal.yml restart data-ingestion
docker-compose -f docker-compose.minimal.yml restart vector-search
docker-compose -f docker-compose.minimal.yml restart training-pipeline

# View logs
docker-compose -f docker-compose.minimal.yml logs -f
docker-compose -f docker-compose.minimal.yml logs -f data-ingestion

# Resource monitoring
docker stats --no-stream
```

## 🌐 Access Your Application

**Replace `localhost` with your server IP (e.g., 206.189.131.217)**

```bash
# Frontend
http://localhost:8090

# Dashboard  
http://localhost:8090/dashboard.html

# Admin Panel
http://localhost:8090/admin/dashboard.html

# Authentication
http://localhost:8090/auth.html
```

## 🔧 Troubleshooting Commands

### **If Services Won't Start**
```bash
# Clean everything and start fresh
./cleanup.sh
./fix-deployment.sh
```

### **If Running Out of Memory**
```bash
# Check memory usage
free -h

# Clean Docker system
docker system prune -f

# Restart services with clean slate
docker-compose -f docker-compose.minimal.yml down
./fix-deployment.sh
```

### **If Containers Keep Crashing**
```bash
# Check logs for errors
docker-compose -f docker-compose.minimal.yml logs -f

# Check specific service
docker logs container_name

# Restart problematic service
docker-compose -f docker-compose.minimal.yml restart service_name
```

## 📊 Health Checks

### **Quick Health Test**
```bash
# Test all services
curl http://localhost:8090/health
curl http://localhost:8091/health  # Data Ingestion
curl http://localhost:8092/health  # Vector Search  
curl http://localhost:8093/health  # Training Pipeline

# Test frontend
curl http://localhost:8090
```

### **Resource Monitoring**
```bash
# System resources
free -h && df -h

# Docker container resources
docker stats --no-stream

# Service status
docker-compose -f docker-compose.minimal.yml ps
```

## 🔄 New Server Deployment

### **Complete Fresh Setup**
```bash
# 1. Clone repository
git clone <your-repo-url> chatbot-saas
cd chatbot-saas

# 2. Setup environment
cp .env.example .env
nano .env  # Edit configuration

# 3. Deploy
./cleanup.sh    # Clean any previous installs
./fix-deployment.sh  # Deploy services

# 4. Verify
curl http://localhost:8090/health
```

### **Migration from Old Server**
```bash
# On old server - backup data
docker exec postgres_container pg_dump -U chatbot_saas_user chatbot_saas_db > backup.sql

# On new server - restore data  
docker exec new_postgres_container psql -U chatbot_saas_user -d chatbot_saas_db < backup.sql
```

## ⚠️ Emergency Commands

### **Service Down Emergency**
```bash
# Quick restart everything
docker-compose -f docker-compose.minimal.yml restart

# If that fails, nuclear option:
./cleanup.sh
./fix-deployment.sh
```

### **Out of Disk Space**
```bash
# Clean Docker system
docker system prune -af
docker volume prune -f

# Clean logs
sudo truncate -s 0 /var/log/syslog
docker-compose -f docker-compose.minimal.yml down
./fix-deployment.sh
```

### **High Memory Usage**
```bash
# Find memory hogs
docker stats --no-stream

# Restart services
docker-compose -f docker-compose.minimal.yml restart

# Clean and restart
docker system prune -f
./fix-deployment.sh
```

## 🔐 Security & Maintenance

### **SSL Setup (Production)**
```bash
# Install SSL certificate
sudo certbot --nginx -d your-domain.com

# Renew SSL
sudo certbot renew --dry-run
```

### **Backup Commands**
```bash
# Database backup
docker exec postgres_container pg_dump -U chatbot_saas_user chatbot_saas_db > backup_$(date +%Y%m%d).sql

# Full system backup
tar -czf chatbot_backup_$(date +%Y%m%d).tar.gz /opt/chatbot-saas
```

### **Update Deployment**
```bash
# Pull latest changes
git pull origin main

# Rebuild and deploy
docker-compose -f docker-compose.minimal.yml down
./fix-deployment.sh
```

## 📈 Scaling Commands

### **Upgrade Server Resources**
```bash
# Check current usage
free -h
df -h
docker stats --no-stream

# For 8GB server upgrade
./cleanup.sh
# Edit docker-compose.minimal.yml to increase memory limits
./fix-deployment.sh
```

## 🎯 One-Liner Commands

```bash
# Complete restart
./cleanup.sh && ./fix-deployment.sh

# Quick status check
docker-compose -f docker-compose.minimal.yml ps && curl -s http://localhost:8090/health

# Resource overview
free -h && df -h && docker stats --no-stream

# Service logs
docker-compose -f docker-compose.minimal.yml logs -f --tail=50

# Emergency clean restart
docker-compose -f docker-compose.minimal.yml down && docker system prune -f && ./fix-deployment.sh
```

---

**💡 Pro Tip**: Bookmark this file and always use `./fix-deployment.sh` as your go-to deployment command. It's safe and preserves your existing services!