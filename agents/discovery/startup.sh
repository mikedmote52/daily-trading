#!/bin/bash
# Startup script for Render deployment with MCP optimization
# This runs before the main application starts

set -e

echo "🚀 EXPLOSIVE DISCOVERY API STARTUP"
echo "=================================="

# Check environment
echo "📊 Environment: $ENVIRONMENT"
echo "🌐 Render: ${RENDER:-false}"

# Ensure MCP is available if possible
if [ "$RENDER" = "true" ]; then
    echo "🔧 Checking MCP installation on Render..."

    # Check Polygon MCP availability
    if [ ! -z "$POLYGON_API_KEY" ]; then
        echo "📡 Checking Polygon MCP package..."

        # Test if mcp_polygon package is available
        if python3 -c "import mcp_polygon; print('✅ MCP Polygon package available')" 2>/dev/null; then
            echo "✅ Polygon MCP package is available"
        else
            echo "⚠️  Polygon MCP package not available - using HTTP fallback"
        fi
    else
        echo "⚠️  No POLYGON_API_KEY - MCP functionality disabled"
    fi
fi

# Test Python environment
echo "🐍 Testing Python environment..."
python3 -c "
import sys
print(f'Python version: {sys.version}')
print('Discovery system: ✅ Ready (test skipped for faster startup)')
"

echo "=================================="
echo "✅ Startup checks complete"
echo "🚀 Starting Discovery API..."

# Execute the main command passed to this script
exec "$@"