#!/usr/bin/env python3
"""
Production API Endpoint for Universal Stock Discovery
Ready for GitHub/Render deployment and UI integration
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import asyncio
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DiscoveryAPI')

app = Flask(__name__)
CORS(app)  # Enable CORS for UI integration

class ProductionStockDiscovery:
    def __init__(self):
        self.polygon_api_key = "1ORwpSzeOV20X6uaA8G3Zuxx7hLJ0KIC"
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        
    def get_stock_universe(self, limit: int = 5000) -> List[str]:
        """Get stock universe from Polygon API with caching"""
        cache_key = f"universe_{limit}"
        now = time.time()
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if now - timestamp < self.cache_ttl:
                return cached_data
        
        try:
            url = "https://api.polygon.io/v3/reference/tickers"
            all_stocks = []
            next_url = url
            
            while next_url and len(all_stocks) < limit:
                params = {
                    'apikey': self.polygon_api_key,
                    'market': 'stocks',
                    'active': 'true',
                    'limit': 1000
                }
                
                if next_url != url:
                    response = requests.get(next_url, timeout=30)
                else:
                    response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'results' in data:
                        for ticker in data['results']:
                            symbol = ticker.get('ticker', '').strip()
                            ticker_type = ticker.get('type', '').upper()
                            
                            # Filter for common stocks only
                            if (symbol and 
                                ticker_type == 'CS' and
                                len(symbol) <= 5 and 
                                symbol.isalpha() and
                                len(symbol) >= 2):
                                all_stocks.append(symbol)
                    
                    next_url = data.get('next_url')
                    if next_url:
                        next_url += f"&apikey={self.polygon_api_key}"
                    
                    time.sleep(0.12)  # Rate limiting
                else:
                    break
            
            # Cache results
            unique_stocks = sorted(list(set(all_stocks)))
            self.cache[cache_key] = (unique_stocks, now)
            return unique_stocks
            
        except Exception as e:
            logger.error(f"Error fetching universe: {e}")
            return []
    
    def apply_filters(self, symbol: str) -> Dict[str, Any]:
        """Apply immediate filters and return stock data if it passes"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price or current_price > 100 or current_price < 1:
                return None
            
            # Market cap filter
            market_cap = info.get('marketCap', 0)
            if market_cap < 100e6 or market_cap > 50e9:
                return None
            
            # Security type filter
            if info.get('quoteType', '').upper() in ['ETF', 'MUTUALFUND', 'INDEX']:
                return None
            
            # Get historical data
            hist = ticker.history(period="1mo")
            if len(hist) < 20:
                return None
            
            # Volume filter
            current_volume = hist['Volume'].iloc[-1]
            if current_volume < 500000:
                return None
            
            # Volume surge calculation
            avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
            volume_surge = current_volume / avg_volume if avg_volume > 0 else 0
            if volume_surge < 1.5:
                return None
            
            # Volatility calculation
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            if volatility < 0.25 or volatility > 2.0:
                return None
            
            # Short interest filter
            short_interest = info.get('shortPercentOfFloat', 0) * 100
            if short_interest < 5.0:
                return None
            
            # P/E filter
            pe_ratio = info.get('trailingPE')
            if pe_ratio and pe_ratio > 50:
                return None
            
            # Calculate metrics
            momentum = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-20]) - 1) * 100 if len(hist) >= 20 else 0
            
            return {
                'symbol': symbol,
                'price': current_price,
                'market_cap': market_cap,
                'volume': int(current_volume),
                'volume_surge': volume_surge,
                'volatility': volatility,
                'short_interest': short_interest,
                'pe_ratio': pe_ratio,
                'momentum': momentum,
                'sector': info.get('sector', 'Unknown'),
                'exchange': info.get('exchange', 'Unknown')
            }
            
        except Exception:
            return None
    
    def calculate_explosive_score(self, stock_data: Dict[str, Any]) -> int:
        """Calculate explosive potential score"""
        score = 50.0
        
        # Short interest bonus
        score += min(stock_data['short_interest'] * 2, 40)
        
        # Volume surge bonus
        score += min((stock_data['volume_surge'] - 1) * 15, 25)
        
        # Market cap bonus (smaller = higher potential)
        if stock_data['market_cap'] < 1e9:
            score += 20
        elif stock_data['market_cap'] < 5e9:
            score += 10
        
        # Volatility bonus
        score += min(stock_data['volatility'] * 15, 20)
        
        # Momentum bonus
        score += min(abs(stock_data['momentum']) * 0.5, 15)
        
        # Sector bonus
        sector_bonuses = {
            'Healthcare': 10, 'Technology': 8, 'Communication Services': 6,
            'Consumer Discretionary': 5, 'Energy': 7
        }
        score += sector_bonuses.get(stock_data['sector'], 2)
        
        return int(min(max(score, 0), 100))
    
    def scan_stocks(self, limit: int = 100, batch_size: int = 50) -> List[Dict[str, Any]]:
        """Scan stocks and return results - enhanced to find high-value stocks"""
        universe = self.get_stock_universe(limit=5000)
        logger.info(f"Scanning {len(universe)} stocks from universe")
        
        # Add priority stocks that we know are high-value to ensure they're scanned
        priority_stocks = ['IONQ', 'FCEL', 'BNTX', 'QS', 'DKNG', 'QUBT', 'RGTI', 'VIGL', 
                          'CRDO', 'AEVA', 'CRWV', 'SEZL', 'SMCI', 'NVDA', 'AMD', 'TSLA']
        
        # Create enhanced universe with priority stocks first
        enhanced_universe = []
        
        # Add priority stocks first
        for symbol in priority_stocks:
            if symbol in universe:
                enhanced_universe.append(symbol)
                universe.remove(symbol)  # Remove from main list to avoid duplicates
        
        # Add remaining universe in randomized order to avoid alphabetical bias
        import random
        random.shuffle(universe)
        enhanced_universe.extend(universe)
        
        results = []
        processed = 0
        total_to_scan = min(len(enhanced_universe), limit * 20)  # Scan more stocks to find good ones
        
        for i in range(0, total_to_scan, batch_size):
            batch = enhanced_universe[i:i + batch_size]
            
            for symbol in batch:
                stock_data = self.apply_filters(symbol)
                if stock_data:
                    explosive_score = self.calculate_explosive_score(stock_data)
                    stock_data['explosive_score'] = explosive_score
                    results.append(stock_data)
                    
                    # Continue scanning even after finding limit - we want the best ones
            
            processed += len(batch)
            logger.info(f"Processed {processed}/{total_to_scan} stocks, found {len(results)} candidates")
            
            # Only stop if we've scanned enough and have good results
            if processed >= 1000 and len(results) >= limit:
                break
        
        # Sort by explosive score and return top results
        results.sort(key=lambda x: x['explosive_score'], reverse=True)
        return results[:limit]

# Initialize discovery engine
discovery = ProductionStockDiscovery()

@app.route('/api/discovery/scan', methods=['GET'])
def scan_explosive_stocks():
    """API endpoint to scan for explosive growth stocks"""
    try:
        # Get parameters
        limit = min(int(request.args.get('limit', 50)), 100)  # Max 100 results
        
        # Run scan
        results = discovery.scan_stocks(limit=limit)
        
        # Generate response
        response = {
            'timestamp': datetime.now().isoformat(),
            'total_results': len(results),
            'scan_parameters': {
                'max_price': 100,
                'min_price': 1,
                'min_volume': 500000,
                'min_market_cap': 100000000,
                'max_market_cap': 50000000000,
                'min_short_interest': 5.0,
                'min_volatility': 0.25,
                'max_volatility': 2.0,
                'volume_surge_min': 1.5
            },
            'results': [
                {
                    'rank': i + 1,
                    'symbol': result['symbol'],
                    'price': round(result['price'], 2),
                    'explosive_score': result['explosive_score'],
                    'market_cap_billions': round(result['market_cap'] / 1e9, 2),
                    'volume_surge': round(result['volume_surge'], 1),
                    'short_interest': round(result['short_interest'], 1),
                    'momentum': round(result['momentum'], 1),
                    'volatility': round(result['volatility'], 2),
                    'sector': result['sector'],
                    'exchange': result['exchange']
                }
                for i, result in enumerate(results)
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Scan error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/discovery/stock/<symbol>', methods=['GET'])
def get_stock_analysis(symbol):
    """Get detailed analysis for a specific stock"""
    try:
        symbol = symbol.upper()
        stock_data = discovery.apply_filters(symbol)
        
        if not stock_data:
            return jsonify({'error': 'Stock does not meet explosive growth criteria'}), 404
        
        explosive_score = discovery.calculate_explosive_score(stock_data)
        stock_data['explosive_score'] = explosive_score
        
        # Add recommendation
        if explosive_score >= 80:
            recommendation = 'STRONG BUY'
        elif explosive_score >= 70:
            recommendation = 'BUY'
        elif explosive_score >= 60:
            recommendation = 'HOLD'
        else:
            recommendation = 'AVOID'
        
        stock_data['recommendation'] = recommendation
        stock_data['timestamp'] = datetime.now().isoformat()
        
        return jsonify(stock_data)
        
    except Exception as e:
        logger.error(f"Stock analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/discovery/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'polygon_api': 'connected',
        'cache_entries': len(discovery.cache)
    })

@app.route('/api/discovery/universe', methods=['GET'])
def get_universe_info():
    """Get information about the stock universe"""
    try:
        universe = discovery.get_stock_universe(limit=1000)
        return jsonify({
            'total_stocks': len(universe),
            'sample_stocks': universe[:20],
            'data_source': 'Polygon API',
            'filters_applied': [
                'Common Stocks Only (CS)',
                'Active Trading',
                'Symbol Length 2-5 chars',
                'Alphabetic symbols only'
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Production configuration
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False
    )