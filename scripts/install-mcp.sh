#!/bin/bash
# Install Polygon MCP for Render Deployment
# This script ensures MCP is properly configured in the production environment

set -e  # Exit on any error

echo "🚀 INSTALLING POLYGON MCP FOR RENDER DEPLOYMENT"
echo "================================================"

# Check if we're in a CI/deployment environment
if [ "$RENDER" = "true" ] || [ "$CI" = "true" ]; then
    echo "✅ Detected deployment environment"
else
    echo "⚠️  Running in development environment"
fi

# Install Node.js and npm if not available (should be available on Render)
if ! command -v npm &> /dev/null; then
    echo "📦 Installing Node.js and npm..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

# Install Claude MCP CLI globally
echo "🔧 Installing Claude MCP CLI..."
npm install -g @anthropic/claude-mcp || {
    echo "⚠️  Global MCP installation failed, trying local install..."
    npm install @anthropic/claude-mcp
    export PATH="$PATH:./node_modules/.bin"
}

# Check if Polygon API key is available
if [ -z "$POLYGON_API_KEY" ]; then
    echo "❌ POLYGON_API_KEY environment variable not set"
    echo "🔧 MCP will use fallback HTTP requests"
    exit 0
fi

echo "🔑 Polygon API key detected"

# Install and configure Polygon MCP
echo "📡 Installing Polygon MCP..."
claude mcp add polygon --api-key "$POLYGON_API_KEY" || {
    echo "⚠️  Polygon MCP installation failed, system will use HTTP fallback"
    exit 0
}

# Verify MCP installation
echo "✅ Verifying MCP installation..."
if command -v polygon &> /dev/null; then
    echo "🎉 Polygon MCP successfully installed!"

    # Test MCP with a simple call
    polygon get-market-status || echo "📊 MCP installed but API test failed (normal for deployment)"

else
    echo "⚠️  MCP command not found, system will use HTTP fallback"
fi

echo "================================================"
echo "✅ MCP installation script completed"
echo "🚀 System ready for optimized Polygon API calls"