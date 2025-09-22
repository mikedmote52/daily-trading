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

    # Try to configure MCP if not already done
    if [ ! -z "$POLYGON_API_KEY" ] && ! command -v polygon &> /dev/null; then
        echo "📡 Attempting to configure Polygon MCP..."

        # Install MCP if npm is available
        if command -v npm &> /dev/null; then
            npm install -g @anthropic/claude-mcp --silent || echo "⚠️  MCP CLI installation failed"

            # Configure Polygon MCP
            claude mcp add polygon --api-key "$POLYGON_API_KEY" &>/dev/null || echo "⚠️  MCP configuration failed"
        fi
    fi

    # Verify MCP status
    if command -v polygon &> /dev/null; then
        echo "✅ Polygon MCP is available"
    else
        echo "⚠️  Polygon MCP not available - using HTTP fallback"
    fi
fi

# Test Python environment
echo "🐍 Testing Python environment..."
python3 -c "
import sys
print(f'Python version: {sys.version}')

# Test discovery system import
try:
    from universal_discovery import UniversalDiscoverySystem
    print(f'Discovery system: ✅ Ready')
except Exception as e:
    print(f'Discovery system: ❌ {e}')
    sys.exit(1)
"

echo "=================================="
echo "✅ Startup checks complete"
echo "🚀 Starting Discovery API..."

# Execute the main command passed to this script
exec "$@"