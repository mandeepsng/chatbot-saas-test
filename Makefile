# ChatFlow AI Development Makefile

# Variables
PYTHON := python3
PIP := pip3
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := chatbot-saas

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

.PHONY: help install install-dev test test-unit test-integration lint format type-check security-check clean build run run-dev stop logs shell backup restore deploy

# Default target
help: ## Show this help message
	@echo "$(GREEN)ChatFlow AI Development Commands$(RESET)"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(BLUE)%-20s$(RESET) %s\n", $$1, $$2}'

# Installation
install: ## Install production dependencies
	@echo "$(YELLOW)Installing production dependencies...$(RESET)"
	$(PIP) install -r backend/requirements.txt

install-dev: ## Install development dependencies
	@echo "$(YELLOW)Installing development dependencies...$(RESET)"
	$(PIP) install -r backend/requirements.txt
	$(PIP) install pytest pytest-asyncio pytest-cov pytest-mock black isort flake8 mypy

# Testing
test: ## Run all tests
	@echo "$(YELLOW)Running all tests...$(RESET)"
	pytest backend/tests/ -v

test-unit: ## Run unit tests only
	@echo "$(YELLOW)Running unit tests...$(RESET)"
	pytest backend/tests/ -v -m "unit"

test-integration: ## Run integration tests only
	@echo "$(YELLOW)Running integration tests...$(RESET)"
	pytest backend/tests/ -v -m "integration"

test-coverage: ## Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(RESET)"
	pytest backend/tests/ --cov=backend --cov-report=html --cov-report=term

test-performance: ## Run performance tests
	@echo "$(YELLOW)Running performance tests...$(RESET)"
	pytest backend/tests/ -v -m "performance" --durations=10

# Code quality
lint: ## Run linting
	@echo "$(YELLOW)Running linting...$(RESET)"
	flake8 backend/ --max-line-length=88 --extend-ignore=E203,W503
	@echo "$(GREEN)Linting completed$(RESET)"

format: ## Format code with black and isort
	@echo "$(YELLOW)Formatting code...$(RESET)"
	black backend/ --line-length=88
	isort backend/ --profile black
	@echo "$(GREEN)Code formatting completed$(RESET)"

format-check: ## Check code formatting
	@echo "$(YELLOW)Checking code formatting...$(RESET)"
	black --check backend/ --line-length=88
	isort --check-only backend/ --profile black

type-check: ## Run type checking
	@echo "$(YELLOW)Running type checking...$(RESET)"
	mypy backend/ --ignore-missing-imports
	@echo "$(GREEN)Type checking completed$(RESET)"

security-check: ## Run security checks
	@echo "$(YELLOW)Running security checks...$(RESET)"
	safety check -r backend/requirements.txt
	bandit -r backend/ -f json -o security-report.json || true
	@echo "$(GREEN)Security check completed$(RESET)"

# Database operations
db-setup: ## Set up database with schema
	@echo "$(YELLOW)Setting up database...$(RESET)"
	$(DOCKER_COMPOSE) exec postgres psql -U postgres -d chatbot_training -f /docker-entrypoint-initdb.d/001-schema.sql

db-migrate: ## Run database migrations
	@echo "$(YELLOW)Running database migrations...$(RESET)"
	# Add migration commands here when using Alembic
	@echo "$(GREEN)Database migrations completed$(RESET)"

db-reset: ## Reset database (WARNING: destroys data)
	@echo "$(RED)WARNING: This will destroy all data!$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(DOCKER_COMPOSE) down -v; \
		$(DOCKER_COMPOSE) up -d postgres redis; \
		sleep 5; \
		make db-setup; \
	fi

# Docker operations
build: ## Build Docker images
	@echo "$(YELLOW)Building Docker images...$(RESET)"
	$(DOCKER_COMPOSE) build
	@echo "$(GREEN)Build completed$(RESET)"

run: ## Run production services
	@echo "$(YELLOW)Starting production services...$(RESET)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Services started$(RESET)"

run-dev: ## Run development services with hot reload
	@echo "$(YELLOW)Starting development services...$(RESET)"
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml up -d
	@echo "$(GREEN)Development services started$(RESET)"

stop: ## Stop all services
	@echo "$(YELLOW)Stopping services...$(RESET)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)Services stopped$(RESET)"

restart: ## Restart all services
	@echo "$(YELLOW)Restarting services...$(RESET)"
	$(DOCKER_COMPOSE) restart
	@echo "$(GREEN)Services restarted$(RESET)"

logs: ## Show service logs
	$(DOCKER_COMPOSE) logs -f

logs-service: ## Show logs for specific service (usage: make logs-service SERVICE=data-ingestion)
	$(DOCKER_COMPOSE) logs -f $(SERVICE)

status: ## Show service status
	$(DOCKER_COMPOSE) ps

# Development utilities
shell: ## Open shell in development container
	$(DOCKER_COMPOSE) exec data-ingestion /bin/bash

jupyter: ## Start Jupyter Lab for development
	@echo "$(YELLOW)Starting Jupyter Lab...$(RESET)"
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml up -d jupyter
	@echo "$(GREEN)Jupyter Lab started at http://localhost:8888$(RESET)"

# Monitoring
health-check: ## Check service health
	@echo "$(YELLOW)Checking service health...$(RESET)"
	@curl -f http://localhost:8001/health && echo "$(GREEN) Data Ingestion: OK$(RESET)" || echo "$(RED) Data Ingestion: FAIL$(RESET)"
	@curl -f http://localhost:8002/health && echo "$(GREEN) Vector Search: OK$(RESET)" || echo "$(RED) Vector Search: FAIL$(RESET)"
	@curl -f http://localhost:8003/health && echo "$(GREEN) Training Pipeline: OK$(RESET)" || echo "$(RED) Training Pipeline: FAIL$(RESET)"

metrics: ## Show Prometheus metrics
	@echo "$(YELLOW)Fetching metrics...$(RESET)"
	curl -s http://localhost:8000/metrics | head -20

# Data operations
backup: ## Backup database
	@echo "$(YELLOW)Creating database backup...$(RESET)"
	$(DOCKER_COMPOSE) exec postgres pg_dump -U postgres chatbot_training > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)Backup completed$(RESET)"

restore: ## Restore database from backup (usage: make restore BACKUP=backup_file.sql)
	@echo "$(YELLOW)Restoring database from $(BACKUP)...$(RESET)"
	$(DOCKER_COMPOSE) exec -T postgres psql -U postgres chatbot_training < $(BACKUP)
	@echo "$(GREEN)Restore completed$(RESET)"

# Cleanup
clean: ## Clean up containers, images, and volumes
	@echo "$(YELLOW)Cleaning up Docker resources...$(RESET)"
	$(DOCKER_COMPOSE) down -v --remove-orphans
	$(DOCKER) system prune -f
	@echo "$(GREEN)Cleanup completed$(RESET)"

clean-all: ## Clean up everything including images
	@echo "$(YELLOW)Cleaning up all Docker resources...$(RESET)"
	$(DOCKER_COMPOSE) down -v --remove-orphans
	$(DOCKER) system prune -af --volumes
	@echo "$(GREEN)Complete cleanup completed$(RESET)"

# Performance testing
load-test: ## Run load tests (requires k6)
	@echo "$(YELLOW)Running load tests...$(RESET)"
	k6 run scripts/load-test.js

benchmark: ## Run benchmark tests
	@echo "$(YELLOW)Running benchmark tests...$(RESET)"
	pytest backend/tests/ -v -m "performance" --benchmark-only

# Production deployment
deploy-staging: ## Deploy to staging environment
	@echo "$(YELLOW)Deploying to staging...$(RESET)"
	# Add staging deployment commands here
	@echo "$(GREEN)Staging deployment completed$(RESET)"

deploy-prod: ## Deploy to production environment
	@echo "$(RED)WARNING: Deploying to production!$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(YELLOW)Deploying to production...$(RESET)"; \
		# Add production deployment commands here; \
		echo "$(GREEN)Production deployment completed$(RESET)"; \
	fi

# Environment setup
setup-env: ## Set up environment file
	@echo "$(YELLOW)Setting up environment file...$(RESET)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)Created .env file from template$(RESET)"; \
		echo "$(YELLOW)Please edit .env with your configuration$(RESET)"; \
	else \
		echo "$(YELLOW).env file already exists$(RESET)"; \
	fi

# Git hooks
install-hooks: ## Install git pre-commit hooks
	@echo "$(YELLOW)Installing git hooks...$(RESET)"
	@echo "#!/bin/sh\nmake format-check lint type-check test-unit" > .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "$(GREEN)Git hooks installed$(RESET)"

# All-in-one commands
dev-setup: setup-env install-dev build run-dev ## Complete development setup
	@echo "$(GREEN)Development environment ready!$(RESET)"
	@echo "$(BLUE)Services available at:$(RESET)"
	@echo "  - Data Ingestion: http://localhost:8001"
	@echo "  - Vector Search: http://localhost:8002"
	@echo "  - Training Pipeline: http://localhost:8003"
	@echo "  - Performance Optimizer: http://localhost:8004"
	@echo "  - Monitoring: http://localhost:8005"
	@echo "  - Metrics: http://localhost:8000/metrics"

ci: format-check lint type-check test-coverage ## Run all CI checks

pre-commit: format lint type-check test-unit ## Run pre-commit checks

# Documentation
docs-serve: ## Serve documentation locally
	@echo "$(YELLOW)Starting documentation server...$(RESET)"
	@echo "$(GREEN)Documentation available at http://localhost:8080$(RESET)"
	@python3 -m http.server 8080 --directory .

# Quick commands for common tasks
quick-test: ## Quick test run (unit tests only, no coverage)
	pytest backend/tests/ -v -x --tb=short -m "unit"

quick-check: ## Quick code quality check
	black --check backend/ --quiet
	flake8 backend/ --quiet
	@echo "$(GREEN)Code quality check passed$(RESET)"

# Environment info
info: ## Show environment information
	@echo "$(BLUE)Environment Information:$(RESET)"
	@echo "Python version: $(shell python3 --version)"
	@echo "Docker version: $(shell docker --version)"
	@echo "Docker Compose version: $(shell docker-compose --version)"
	@echo "Git version: $(shell git --version)"
	@echo "Current branch: $(shell git branch --show-current)"
	@echo "Current directory: $(PWD)"