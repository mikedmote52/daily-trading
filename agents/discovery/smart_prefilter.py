#!/usr/bin/env python3
"""
Smart Pre-filtering System

Strategy:
1. Get all stock symbols from Polygon (fast)
2. Use Polygon aggregates API to get price data in batches (fast)
3. Immediately filter by price < $100 
4. Only run Yahoo Finance on price-filtered candidates
5. This reduces expensive API calls from 5000+ to ~500
"""

import requests
import time
import logging
from typing import Dict, List, Any, Optional
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SmartPrefilter')

class SmartPreFilterScanner:
    def __init__(self):
        self.polygon_api_key = "1ORwpSzeOV20X6uaA8G3Zuxx7hLJ0KIC"
        
    def get_all_symbols(self) -> List[str]:
        """Get all stock symbols quickly"""
        logger.info("📋 Fetching all stock symbols...")
        
        url = "https://api.polygon.io/v3/reference/tickers"
        all_symbols = []
        next_url = url
        
        while next_url:
            params = {
                'apikey': self.polygon_api_key,
                'market': 'stocks',
                'active': 'true',
                'type': 'CS',  # Common Stock only
                'limit': 1000
            }
            
            try:
                if next_url != url:
                    response = requests.get(next_url, timeout=30)
                else:
                    response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'results' in data:
                        for ticker in data['results']:
                            symbol = ticker.get('ticker', '').strip()
                            if (symbol and len(symbol) <= 5 and symbol.isalpha() and len(symbol) >= 2):
                                all_symbols.append(symbol)
                    
                    next_url = data.get('next_url')
                    if next_url:
                        next_url += f"&apikey={self.polygon_api_key}"
                    
                    logger.info(f"Collected {len(all_symbols)} symbols...")
                    time.sleep(0.12)
                    
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Error: {e}")
                break
        
        logger.info(f"✅ Collected {len(all_symbols)} total symbols")
        return all_symbols
    
    def get_prices_batch(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for a batch of symbols using Polygon"""
        batch_prices = {}
        
        # Use previous close prices from Polygon (much faster than real-time)
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        for symbol in symbols:
            try:
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{yesterday}/{yesterday}"
                params = {'apikey': self.polygon_api_key}
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'results' in data and data['results']:
                        close_price = data['results'][0]['c']  # Close price
                        batch_prices[symbol] = close_price
                
                time.sleep(0.05)  # Rate limiting
                
            except Exception:
                continue
        
        return batch_prices
    
    def price_prefilter(self, symbols: List[str], batch_size: int = 50) -> List[str]:
        """Pre-filter symbols by price using Polygon data"""
        logger.info(f"💰 Pre-filtering {len(symbols)} symbols by price...")
        
        under_100_symbols = []
        processed = 0
        
        # Process in batches to manage rate limits
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_prices = self.get_prices_batch(batch)
            
            # Filter by price
            for symbol, price in batch_prices.items():
                if 0.01 <= price <= 100.0:  # Price filter
                    under_100_symbols.append(symbol)
            
            processed += len(batch)
            logger.info(f"Processed {processed}/{len(symbols)}, found {len(under_100_symbols)} under $100")
            
            if processed >= 1000:  # Limit for testing
                break
        
        logger.info(f"✅ Price pre-filtering: {len(under_100_symbols)} stocks under $100")
        return under_100_symbols
    
    def detailed_filter_batch(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Apply detailed filtering using Yahoo Finance on pre-filtered symbols"""
        logger.info(f"🔍 Detailed filtering on {len(symbols)} price-filtered candidates...")
        
        final_candidates = []
        
        for i, symbol in enumerate(symbols):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get current price
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                if not current_price or current_price > 100 or current_price < 0.01:
                    continue
                
                # Market cap filter
                market_cap = info.get('marketCap', 0)
                if market_cap < 100e6 or market_cap > 50e9:
                    continue
                
                # Security type filter
                if info.get('quoteType', '').upper() in ['ETF', 'MUTUALFUND', 'INDEX']:
                    continue
                
                # Historical data
                hist = ticker.history(period="1mo")
                if len(hist) < 20:
                    continue
                
                # Volume filter
                current_volume = hist['Volume'].iloc[-1]
                if current_volume < 500000:
                    continue
                
                # Volume surge
                avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
                volume_surge = current_volume / avg_volume if avg_volume > 0 else 0
                if volume_surge < 1.5:
                    continue
                
                # Volatility
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)
                if volatility < 0.25 or volatility > 2.0:
                    continue
                
                # Short interest
                short_interest = info.get('shortPercentOfFloat', 0) * 100
                if short_interest < 5.0:
                    continue
                
                # P/E ratio
                pe_ratio = info.get('trailingPE')
                if pe_ratio and pe_ratio > 50:
                    continue
                
                # Calculate metrics
                momentum = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-20]) - 1) * 100 if len(hist) >= 20 else 0
                explosive_score = self._calculate_explosive_score({
                    'price': current_price,
                    'market_cap': market_cap,
                    'volume_surge': volume_surge,
                    'volatility': volatility,
                    'short_interest': short_interest,
                    'momentum': momentum,
                    'sector': info.get('sector', 'Unknown')
                })
                
                final_candidates.append({
                    'symbol': symbol,
                    'price': current_price,
                    'market_cap': market_cap,
                    'volume': int(current_volume),
                    'volume_surge': volume_surge,
                    'volatility': volatility,
                    'short_interest': short_interest,
                    'pe_ratio': pe_ratio,
                    'momentum': momentum,
                    'explosive_score': explosive_score,
                    'sector': info.get('sector', 'Unknown'),
                    'exchange': info.get('exchange', 'Unknown')
                })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(symbols)}, found {len(final_candidates)} explosive candidates")
                
            except Exception:
                continue
        
        # Sort by explosive score
        final_candidates.sort(key=lambda x: x['explosive_score'], reverse=True)
        return final_candidates
    
    def _calculate_explosive_score(self, stock_data: Dict[str, Any]) -> int:
        """Calculate explosive potential score"""
        score = 50.0
        
        # Short interest bonus
        score += min(stock_data['short_interest'] * 2, 40)
        
        # Volume surge bonus
        score += min((stock_data['volume_surge'] - 1) * 15, 25)
        
        # Small cap bonus
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
    
    def smart_scan(self) -> List[Dict[str, Any]]:
        """Complete smart scanning process"""
        logger.info("🚀 Starting SMART universal scan...")
        
        # Step 1: Get all symbols (fast)
        all_symbols = self.get_all_symbols()
        
        # Step 2: Pre-filter by price (reduces candidates by ~80%)
        price_filtered = self.price_prefilter(all_symbols)
        
        # Step 3: Detailed filtering on price-filtered candidates only
        final_candidates = self.detailed_filter_batch(price_filtered)
        
        logger.info("📊 SMART SCAN SUMMARY:")
        logger.info(f"   Total Universe: {len(all_symbols)} stocks")
        logger.info(f"   Price-filtered: {len(price_filtered)} stocks under $100")
        logger.info(f"   Final explosive candidates: {len(final_candidates)}")
        reduction = ((len(all_symbols) - len(price_filtered)) / len(all_symbols)) * 100
        logger.info(f"   API call reduction: {reduction:.1f}%")
        
        return final_candidates

def main():
    """Test smart scanner"""
    scanner = SmartPreFilterScanner()
    results = scanner.smart_scan()
    
    print("\n🚀 SMART UNIVERSAL SCAN RESULTS")
    print("=" * 60)
    print(f"📊 Total Explosive Candidates: {len(results)}")
    
    if results:
        print("\n🏆 TOP 10 EXPLOSIVE CANDIDATES:")
        for i, stock in enumerate(results[:10], 1):
            print(f"{i:2d}. {stock['symbol']:6s} - ${stock['price']:7.2f} - Score: {stock['explosive_score']:3d}/100")
            print(f"    💰 MCap: ${stock['market_cap']/1e9:4.1f}B | 🔊 Vol: {stock['volume_surge']:4.1f}x | ⚡ SI: {stock['short_interest']:4.1f}%")
    
    return results

if __name__ == "__main__":
    main()