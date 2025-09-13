#!/bin/bash
# Setup script for Daily Trading System

set -e

echo "🚀 Setting up Daily Trading System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed"
    exit 1
fi

# Check if Redis is installed
if ! command -v redis-server &> /dev/null; then
    echo "⚠️  Redis not found. Please install Redis server"
    echo "   macOS: brew install redis"
    echo "   Ubuntu: sudo apt-get install redis-server"
    echo "   Or use Docker: docker run -d -p 6379:6379 redis:latest"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create Python virtual environments for each agent
echo "📦 Setting up Python virtual environments..."

AGENTS=("master" "backend" "discovery" "backtesting" "portfolio")

for agent in "${AGENTS[@]}"; do
    echo "   Setting up $agent agent..."
    cd agents/$agent
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    deactivate
    
    cd ../..
done

# Set up frontend
echo "📦 Setting up frontend..."
cd agents/frontend
npm install
cd ../..

# Install main package dependencies
echo "📦 Installing main package dependencies..."
npm install

# Create .env file from template
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys and configuration"
fi

# Start Redis if not running
echo "🔧 Checking Redis..."
if ! pgrep -x "redis-server" > /dev/null; then
    echo "   Starting Redis server..."
    redis-server --daemonize yes
    sleep 2
fi

# Test Redis connection
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is running"
else
    echo "❌ Redis connection failed"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Edit .env file with your API keys"
echo "   2. Start all agents: npm run dev"
echo "   3. Open http://localhost:3000 in your browser"
echo ""
echo "🔧 Individual agent commands:"
echo "   • Master:      npm run dev:master"
echo "   • Frontend:    npm run dev:frontend"  
echo "   • Backend:     npm run dev:backend"
echo "   • Discovery:   npm run dev:discovery"
echo "   • Backtesting: npm run dev:backtesting"
echo "   • Portfolio:   npm run dev:portfolio"
echo ""