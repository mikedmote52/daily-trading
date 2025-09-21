#!/usr/bin/env python3
"""
MCP-Enhanced Stock Discovery Backend
Serves real stock data filtered by Polygon MCP with accumulation-based scoring
"""
import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
import urllib.parse

# Add the discovery directory to the path
sys.path.append('/Users/michaelmote/Desktop/Daily-Trading/agents/discovery')

# Set environment variables for the discovery system
os.environ['POLYGON_API_KEY'] = '1ORwpSzeOV20X6uaA8G3Zuxx7hLJ0KIC'

try:
    from universal_discovery import UniversalDiscoverySystem
    DISCOVERY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import discovery system: {e}")
    DISCOVERY_AVAILABLE = False

class MCPDiscoveryHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        if DISCOVERY_AVAILABLE:
            self.discovery_system = UniversalDiscoverySystem()
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/stocks/explosive':
            self.handle_explosive_stocks()
        elif self.path == '/health':
            self.handle_health_check()
        else:
            self.send_404()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def handle_explosive_stocks(self):
        try:
            print("🔍 Running MCP-enhanced discovery system...")

            if not DISCOVERY_AVAILABLE:
                self.send_error_response("Discovery system not available", 500)
                return

            # Run the full MCP-enhanced discovery pipeline
            result = self.discovery_system.run_universal_discovery()

            # Transform the results to match frontend expectations
            frontend_stocks = []
            for stock in result.get('results', []):
                # Map the discovery result to frontend format
                frontend_stock = {
                    'symbol': stock.get('symbol', ''),
                    'price': stock.get('price', 0),
                    'score': stock.get('accumulation_score', 0),  # Use accumulation score
                    'volume': stock.get('day_volume', 0),
                    'rvol': stock.get('volume_surge', 1.0),  # Use volume_surge field
                    'change_percent': stock.get('percent_change', 0),
                    'reason': stock.get('thesis', f"{stock.get('symbol', 'Stock')} shows accumulation patterns with {stock.get('volume_surge', 1.0):.1f}x volume surge."),
                    'status': stock.get('status', 'UNKNOWN'),
                    'tier': stock.get('tier', 'UNKNOWN')
                }

                # Add price targets if available
                if 'price_target' in stock:
                    frontend_stock['price_target'] = stock['price_target']
                if 'stop_loss' in stock:
                    frontend_stock['stop_loss'] = stock['stop_loss']

                frontend_stocks.append(frontend_stock)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            response_data = {
                'stocks': frontend_stocks,
                'metadata': {
                    'total_found': len(frontend_stocks),
                    'trade_ready': len([s for s in frontend_stocks if s.get('status') == 'TRADE_READY']),
                    'watchlist': len([s for s in frontend_stocks if s.get('status') == 'WATCHLIST']),
                    'processing_time': result.get('processing_time_seconds', 0),
                    'universe_coverage': result.get('universe_coverage', {}),
                    'timestamp': datetime.now().isoformat(),
                    'mcp_enhanced': True,
                    'filters_applied': 'Under $100, >300K volume, not funds'
                }
            }

            print(f"✅ Returning {len(frontend_stocks)} MCP-filtered stocks")
            response = json.dumps(response_data, indent=2)
            self.wfile.write(response.encode())

        except Exception as e:
            print(f"❌ Error in MCP discovery: {e}")
            import traceback
            traceback.print_exc()
            self.send_error_response(f"Discovery error: {str(e)}", 500)

    def handle_health_check(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        health_data = {
            "status": "healthy",
            "service": "mcp-enhanced-stock-discovery",
            "timestamp": datetime.now().isoformat(),
            "polygon_mcp": "connected" if DISCOVERY_AVAILABLE else "unavailable",
            "discovery_system": "available" if DISCOVERY_AVAILABLE else "unavailable",
            "features": [
                "MCP-enhanced filtering",
                "Under $100 price filter",
                ">300K volume filter",
                "Non-fund filtering",
                "Accumulation-based scoring",
                "Trade-ready classification"
            ]
        }

        response = json.dumps(health_data, indent=2)
        self.wfile.write(response.encode())

    def send_error_response(self, error_message, status_code):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        error_response = {
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
            "service": "mcp-enhanced-stock-discovery"
        }

        response = json.dumps(error_response)
        self.wfile.write(response.encode())

    def send_404(self):
        self.send_response(404)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        error_response = {
            "error": "Endpoint not found",
            "available_endpoints": [
                "GET /stocks/explosive - Get MCP-filtered explosive stock opportunities",
                "GET /health - Service health check"
            ]
        }

        response = json.dumps(error_response)
        self.wfile.write(response.encode())

if __name__ == "__main__":
    port = 8081  # Different port from the old backend
    server = HTTPServer(('localhost', port), MCPDiscoveryHandler)

    print(f"🚀 MCP-Enhanced Stock Discovery Server starting on http://localhost:{port}")
    print(f"📊 Polygon MCP integration: {'✅ Available' if DISCOVERY_AVAILABLE else '❌ Unavailable'}")
    print(f"🎯 Endpoints:")
    print(f"   GET /stocks/explosive - Get MCP-filtered explosive stock opportunities")
    print(f"   GET /health - Service health check")
    print(f"🔧 Features:")
    print(f"   • MCP-enhanced filtering (Under $100, >300K volume, not funds)")
    print(f"   • Accumulation-based scoring")
    print(f"   • Trade-ready classification")
    print(f"   • Real-time Polygon API data")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        server.shutdown()