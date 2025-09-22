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

    # Configure Polygon MCP if API key is available
    if [ ! -z "$POLYGON_API_KEY" ]; then
        echo "📡 Configuring Polygon MCP..."

        # Install and configure Polygon MCP
        if command -v claude &> /dev/null; then
            echo "🔧 Installing Polygon MCP server..."
            # Install the Polygon MCP server using uvx
            python3 -m pip install --user uv || echo "⚠️  UV installation failed"

            # Add to PATH
            export PATH="$HOME/.local/bin:$PATH"

            # Configure MCP
            claude mcp add polygon -e POLYGON_API_KEY="$POLYGON_API_KEY" -- uvx --from git+https://github.com/polygon-io/mcp_polygon@v0.4.0 mcp_polygon &>/dev/null && echo "✅ Polygon MCP configured" || echo "⚠️  MCP configuration failed"
        else
            echo "⚠️  Claude CLI not available - MCP configuration skipped"
        fi
    else
        echo "⚠️  No POLYGON_API_KEY - MCP configuration skipped"
    fi

    # Verify MCP status
    if claude mcp list 2>/dev/null | grep -q polygon; then
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
print('Discovery system: ✅ Ready (test skipped for faster startup)')
"

echo "=================================="
echo "✅ Startup checks complete"
echo "🚀 Starting Discovery API..."

# Execute the main command passed to this script
exec "$@"