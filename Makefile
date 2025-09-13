# Daily Trading System Makefile

.PHONY: help setup install dev clean test docker-up docker-down logs

# Default target
help:
	@echo "Daily Trading System Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup     - Run initial setup script"
	@echo "  make install   - Install all dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make dev       - Start all agents in development mode"
	@echo "  make dev-agent AGENT=master - Start specific agent"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up    - Start all services with Docker Compose"
	@echo "  make docker-down  - Stop all Docker services"
	@echo "  make logs         - View Docker logs"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean     - Clean temporary files and caches"
	@echo "  make test      - Run test suites"
	@echo ""
	@echo "Agent-specific commands:"
	@echo "  make start-master      - Start master agent"
	@echo "  make start-frontend    - Start frontend agent"
	@echo "  make start-backend     - Start backend agent"
	@echo "  make start-discovery   - Start discovery agent"
	@echo "  make start-backtesting - Start backtesting agent"
	@echo "  make start-portfolio   - Start portfolio agent"

# Setup and installation
setup:
	@echo "🚀 Running setup script..."
	@chmod +x scripts/setup.sh
	@./scripts/setup.sh

install: setup
	@echo "📦 Installing dependencies..."
	@npm run install:all

# Development commands
dev:
	@echo "🚀 Starting enhanced system with Claude Code SDK..."
	@npm run dev

dev-standard:
	@echo "🚀 Starting standard system..."
	@npm run dev:standard

dev-enhanced:
	@echo "🚀 Starting enhanced system with inter-agent communication..."
	@npm run dev:enhanced

dev-agent:
	@echo "🚀 Starting $(AGENT) agent..."
	@python3 scripts/start_agent.py $(AGENT)

# Individual agent commands
start-master:
	@python3 scripts/start_agent.py master

start-frontend:
	@python3 scripts/start_agent.py frontend

start-backend:
	@python3 scripts/start_agent.py backend

start-discovery:
	@python3 scripts/start_agent.py discovery

start-backtesting:
	@python3 scripts/start_agent.py backtesting

start-portfolio:
	@python3 scripts/start_agent.py portfolio

# Docker commands
docker-up:
	@echo "🐳 Starting Docker services..."
	@docker-compose up -d

docker-down:
	@echo "🐳 Stopping Docker services..."
	@docker-compose down

docker-build:
	@echo "🐳 Building Docker images..."
	@docker-compose build

logs:
	@echo "📋 Viewing Docker logs..."
	@docker-compose logs -f

# Maintenance
clean:
	@echo "🧹 Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type f -name ".DS_Store" -delete
	@rm -rf agents/frontend/build
	@rm -rf agents/frontend/node_modules/.cache
	@echo "✅ Cleanup complete"

test:
	@echo "🧪 Running tests..."
	@echo "Test framework not yet implemented"

# Redis management
redis-start:
	@echo "🔴 Starting Redis..."
	@redis-server --daemonize yes

redis-stop:
	@echo "🔴 Stopping Redis..."
	@redis-cli shutdown

redis-status:
	@echo "🔴 Redis status:"
	@redis-cli ping

# Environment setup
env-check:
	@echo "🔍 Checking environment..."
	@python3 --version
	@node --version
	@npm --version
	@redis-cli --version || echo "Redis not installed"

# Database commands (if using PostgreSQL)
db-migrate:
	@echo "🗄️  Running database migrations..."
	@echo "Database migration not implemented yet"

db-reset:
	@echo "🗄️  Resetting database..."
	@echo "Database reset not implemented yet"

# Backup and restore
backup:
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@redis-cli --rdb backups/redis-$(shell date +%Y%m%d-%H%M%S).rdb

restore:
	@echo "📥 Restoring from backup..."
	@echo "Restore functionality not implemented yet"

# Security scan
security-scan:
	@echo "🔒 Running security scan..."
	@pip-audit || echo "pip-audit not installed"
	@npm audit || echo "npm audit failed"

# Performance monitoring
monitor:
	@echo "📊 Starting performance monitoring..."
	@echo "Monitoring dashboard not implemented yet"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "Documentation generator not implemented yet"

# Communication monitoring
monitor:
	@echo "🔍 Starting communication monitor..."
	@npm run monitor

monitor-detailed:
	@echo "🔬 Showing detailed communication analysis..."
	@npm run monitor:detailed

# Quick status check
status:
	@echo "📊 System Status:"
	@echo "==================="
	@make redis-status
	@echo ""
	@echo "Agents Status:"
	@echo "  Master:      $(shell pgrep -f "agents/master.*main.py" > /dev/null && echo "✅ Running" || echo "❌ Stopped")"
	@echo "  Frontend:    $(shell pgrep -f "react-scripts start\|npm start" > /dev/null && echo "✅ Running" || echo "❌ Stopped")"
	@echo "  Backend:     $(shell pgrep -f "agents/backend/main.py" > /dev/null && echo "✅ Running" || echo "❌ Stopped")"
	@echo "  Discovery:   $(shell pgrep -f "agents/discovery.*main.py" > /dev/null && echo "✅ Running" || echo "❌ Stopped")"
	@echo "  Backtesting: $(shell pgrep -f "agents/backtesting/main.py" > /dev/null && echo "✅ Running" || echo "❌ Stopped")"
	@echo "  Portfolio:   $(shell pgrep -f "agents/portfolio/main.py" > /dev/null && echo "✅ Running" || echo "❌ Stopped")"
	@echo ""
	@echo "Communication:"
	@echo "  Shared Context: $(shell test -f shared_context/progress.json && echo "✅ Available" || echo "❌ Missing")"
	@echo "  Message Queue:  $(shell test -f shared_context/messages.json && echo "✅ Available" || echo "❌ Missing")"