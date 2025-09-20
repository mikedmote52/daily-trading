#!/bin/bash
echo "🚀 Portfolio API Startup Script"
echo "📁 Current directory: $(pwd)"
echo "📄 Python files available:"
ls -la *.py
echo ""
echo "🔧 Environment variables:"
echo "PORT: ${PORT:-Not Set}"
echo "ALPACA_KEY: ${ALPACA_KEY:+Set}"
echo "ALPACA_SECRET: ${ALPACA_SECRET:+Set}"
echo "ALPACA_BASE_URL: ${ALPACA_BASE_URL:-Not Set}"
echo ""
echo "🎯 Starting portfolio_api.py..."
exec python3 portfolio_api.py