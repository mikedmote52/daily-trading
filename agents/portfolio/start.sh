#!/bin/bash
# Portfolio API Service Startup Script

echo "🚀 Starting Portfolio API Service..."
echo "📊 Real Alpaca Data Only - No Mock Data"

# Check required environment variables
if [ -z "$ALPACA_KEY" ] || [ -z "$ALPACA_SECRET" ]; then
    echo "❌ Error: ALPACA_KEY and ALPACA_SECRET environment variables are required"
    exit 1
fi

echo "✅ Environment variables configured"
echo "🔗 Alpaca Base: ${ALPACA_BASE_URL:-https://paper-api.alpaca.markets}"

# Start the FastAPI service
export PORT=${PORT:-8002}
echo "🌐 Starting on port $PORT..."

python3 portfolio_api.py