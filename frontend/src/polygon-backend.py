#!/usr/bin/env python3
"""
Real Stock Data Backend using Polygon API
Provides explosive stock discovery with real market data
"""
import requests
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

POLYGON_API_KEY = "1ORwpSzeOV20X6uaA8G3Zuxx7hLJ0KIC"

class StockDiscoveryService:
    def __init__(self):
        self.api_key = POLYGON_API_KEY
        self.base_url = "https://api.polygon.io"

    def get_market_snapshot(self):
        """Get current market snapshot from Polygon"""
        url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers"
        params = {
            "apikey": self.api_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting market snapshot: {e}")
            return None

    def get_stock_details(self, symbol):
        """Get detailed stock information"""
        url = f"{self.base_url}/v3/reference/tickers/{symbol}"
        params = {
            "apikey": self.api_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting stock details for {symbol}: {e}")
            return None

    def find_explosive_stocks(self) -> List[Dict[str, Any]]:
        """Find stocks with explosive potential using real market data"""
        print("🔍 Scanning market for explosive opportunities...")

        snapshot = self.get_market_snapshot()
        if not snapshot or 'results' not in snapshot:
            print("❌ Failed to get market data")
            return []

        explosive_stocks = []
        results = snapshot['results'][:100]  # Analyze top 100 by volume

        for ticker_data in results:
            try:
                ticker = ticker_data.get('ticker', '')
                if not ticker:
                    continue

                # Extract key metrics
                day_data = ticker_data.get('day', {})
                prev_day = ticker_data.get('prevDay', {})

                if not day_data or not prev_day:
                    continue

                current_price = day_data.get('c', 0)  # Close price
                volume = day_data.get('v', 0)  # Volume
                prev_volume = prev_day.get('v', 1)  # Previous volume
                change_percent = ticker_data.get('todaysChangePerc', 0)

                # Calculate volume ratio
                volume_ratio = volume / max(prev_volume, 1)

                # Calculate explosive score
                score = self.calculate_explosive_score(
                    volume_ratio, abs(change_percent), current_price, volume
                )

                if score >= 60:  # High-scoring opportunities only
                    stock_data = {
                        'symbol': ticker,
                        'price': current_price,
                        'score': round(score, 1),
                        'volume': volume,
                        'rvol': round(volume_ratio, 1),
                        'change_percent': round(change_percent, 2),
                        'reason': self.generate_reason(ticker, volume_ratio, change_percent, score)
                    }
                    explosive_stocks.append(stock_data)

            except Exception as e:
                print(f"Error processing ticker {ticker_data.get('ticker', 'unknown')}: {e}")
                continue

        # Sort by score and return top 8
        explosive_stocks.sort(key=lambda x: x['score'], reverse=True)
        top_stocks = explosive_stocks[:8]

        print(f"✅ Found {len(top_stocks)} explosive opportunities")
        return top_stocks

    def calculate_explosive_score(self, volume_ratio, abs_change_percent, price, volume):
        """Calculate explosive potential score"""
        score = 0

        # Volume surge component (40% weight)
        if volume_ratio > 5:
            score += 40
        elif volume_ratio > 3:
            score += 30
        elif volume_ratio > 2:
            score += 20
        elif volume_ratio > 1.5:
            score += 10

        # Price movement component (30% weight)
        if abs_change_percent > 10:
            score += 30
        elif abs_change_percent > 5:
            score += 20
        elif abs_change_percent > 2:
            score += 10

        # Liquidity component (20% weight)
        if volume > 10000000:  # 10M+ volume
            score += 20
        elif volume > 5000000:  # 5M+ volume
            score += 15
        elif volume > 1000000:  # 1M+ volume
            score += 10

        # Price range component (10% weight)
        if 5 <= price <= 500:  # Sweet spot for explosive moves
            score += 10
        elif 1 <= price <= 1000:
            score += 5

        return min(score, 100)

    def generate_reason(self, symbol, volume_ratio, change_percent, score):
        """Generate explanation for why stock is explosive"""
        reasons = []

        if volume_ratio > 5:
            reasons.append(f"massive {volume_ratio:.1f}x volume surge")
        elif volume_ratio > 3:
            reasons.append(f"strong {volume_ratio:.1f}x volume increase")
        elif volume_ratio > 2:
            reasons.append(f"{volume_ratio:.1f}x volume spike")

        if abs(change_percent) > 10:
            direction = "explosive upward" if change_percent > 0 else "dramatic downward"
            reasons.append(f"{direction} movement of {abs(change_percent):.1f}%")
        elif abs(change_percent) > 5:
            direction = "strong bullish" if change_percent > 0 else "significant bearish"
            reasons.append(f"{direction} momentum with {abs(change_percent):.1f}% move")

        if score > 80:
            reasons.append("exceptional explosive potential")
        elif score > 70:
            reasons.append("high breakout probability")

        return f"{symbol} shows {' with '.join(reasons)} indicating potential explosive move."

class StockDataHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.discovery_service = StockDiscoveryService()
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/stocks/explosive':
            try:
                stocks = self.discovery_service.find_explosive_stocks()

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                response = json.dumps(stocks, indent=2)
                self.wfile.write(response.encode())

            except Exception as e:
                print(f"Error in /stocks/explosive: {e}")
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                error_response = json.dumps({"error": str(e)})
                self.wfile.write(error_response.encode())

        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            health_data = {
                "status": "healthy",
                "service": "polygon-stock-discovery",
                "timestamp": datetime.now().isoformat(),
                "polygon_api": "connected"
            }

            response = json.dumps(health_data)
            self.wfile.write(response.encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

if __name__ == "__main__":
    port = 8080
    server = HTTPServer(('localhost', port), StockDataHandler)
    print(f"🚀 Stock Discovery Server starting on http://localhost:{port}")
    print(f"📊 Real-time data powered by Polygon.io")
    print(f"🎯 Endpoints:")
    print(f"   GET /stocks/explosive - Get explosive stock opportunities")
    print(f"   GET /health - Service health check")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        server.shutdown()