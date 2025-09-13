#!/usr/bin/env python3
"""
Optimized Universal Stock Scanner

1. Get ENTIRE universe from Polygon
2. Immediately filter by price, market cap, etc. using Polygon data
3. Only run expensive Yahoo Finance calls on pre-filtered candidates
4. This reduces API calls from 5000+ to ~100-200 candidates
"""

import requests
import time
import logging
from typing import Dict, List, Any, Optional
import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('OptimizedScanner')

@dataclass
class PreFilterCriteria:
    """Criteria that can be applied using Polygon data before Yahoo calls"""
    max_price: float = 100.0
    min_price: float = 0.01  # Include penny stocks
    min_market_cap: float = 100e6  # $100M minimum
    max_market_cap: float = 50e9   # $50B maximum

class OptimizedUniversalScanner:
    def __init__(self):
        self.polygon_api_key = "1ORwpSzeOV20X6uaA8G3Zuxx7hLJ0KIC"
        self.pre_filters = PreFilterCriteria()
        
    def get_entire_universe_with_prefiltering(self) -> List[Dict[str, Any]]:
        """Get entire universe and immediately pre-filter using Polygon data"""
        logger.info("🌍 Fetching ENTIRE universe with immediate pre-filtering...")
        
        url = "https://api.polygon.io/v3/reference/tickers"
        all_candidates = []
        next_url = url
        page = 0
        
        while next_url:
            params = {
                'apikey': self.polygon_api_key,
                'market': 'stocks',
                'active': 'true',
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
                            # IMMEDIATE FILTERING - No expensive API calls yet
                            candidate = self._pre_filter_stock(ticker)
                            if candidate:
                                all_candidates.append(candidate)
                    
                    # Pagination
                    next_url = data.get('next_url')
                    if next_url:
                        next_url += f"&apikey={self.polygon_api_key}"
                    
                    page += 1
                    logger.info(f"Page {page}: Found {len(all_candidates)} pre-filtered candidates")
                    time.sleep(0.12)  # Rate limiting
                    
                else:
                    logger.error(f"Polygon API error: {response.status_code}")
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching page {page}: {e}")
                break
        
        logger.info(f"✅ Pre-filtering complete: {len(all_candidates)} candidates from entire universe")
        return all_candidates
    
    def _pre_filter_stock(self, ticker_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Pre-filter using only Polygon data - NO expensive API calls"""
        try:
            symbol = ticker_data.get('ticker', '').strip()
            ticker_type = ticker_data.get('type', '').upper()
            market = ticker_data.get('market', '').upper()
            
            # Basic symbol validation
            if not symbol or len(symbol) > 5 or not symbol.isalpha() or len(symbol) < 2:
                return None
            
            # Must be common stock
            if ticker_type != 'CS' or market != 'STOCKS':
                return None
            
            # Get price and market cap from Polygon if available
            last_quote = ticker_data.get('last_quote')
            market_cap = ticker_data.get('market_cap')
            
            # If we have price data from Polygon, use it for immediate filtering
            if last_quote:
                price = last_quote.get('last')
                if price:
                    # IMMEDIATE PRICE FILTER
                    if price > self.pre_filters.max_price or price < self.pre_filters.min_price:
                        return None
            
            # If we have market cap from Polygon, use it for immediate filtering
            if market_cap:
                # IMMEDIATE MARKET CAP FILTER
                if (market_cap < self.pre_filters.min_market_cap or 
                    market_cap > self.pre_filters.max_market_cap):
                    return None
            
            # Stock passed pre-filtering
            return {
                'symbol': symbol,
                'polygon_price': last_quote.get('last') if last_quote else None,
                'polygon_market_cap': market_cap,
                'sector': ticker_data.get('sic_description', 'Unknown'),
                'exchange': ticker_data.get('primary_exchange', 'Unknown')
            }
            
        except Exception:
            return None
    
    def apply_detailed_filters(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply detailed filtering using Yahoo Finance - only on pre-filtered candidates"""
        logger.info(f"🔍 Running detailed analysis on {len(candidates)} pre-filtered candidates...")
        
        final_candidates = []
        processed = 0
        
        for candidate in candidates:
            try:
                symbol = candidate['symbol']
                
                # Now make the expensive Yahoo Finance call
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get current price (use Yahoo as authoritative)
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                if not current_price:
                    continue
                
                # DOUBLE-CHECK PRICE FILTER (Yahoo data is more reliable)
                if current_price > 100 or current_price < 0.01:
                    continue
                
                # Market cap double-check
                market_cap = info.get('marketCap', 0)
                if market_cap < 100e6 or market_cap > 50e9:
                    continue
                
                # Security type filter
                if info.get('quoteType', '').upper() in ['ETF', 'MUTUALFUND', 'INDEX']:
                    continue
                
                # Get historical data for advanced filtering
                hist = ticker.history(period="1mo")
                if len(hist) < 20:
                    continue
                
                # VOLUME FILTER
                current_volume = hist['Volume'].iloc[-1]
                if current_volume < 500000:
                    continue
                
                # VOLUME SURGE FILTER
                avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
                volume_surge = current_volume / avg_volume if avg_volume > 0 else 0
                if volume_surge < 1.5:
                    continue
                
                # VOLATILITY FILTER
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)
                if volatility < 0.25 or volatility > 2.0:
                    continue
                
                # SHORT INTEREST FILTER
                short_interest = info.get('shortPercentOfFloat', 0) * 100
                if short_interest < 5.0:
                    continue
                
                # P/E FILTER
                pe_ratio = info.get('trailingPE')
                if pe_ratio and pe_ratio > 50:
                    continue
                
                # PASSED ALL FILTERS - Calculate final metrics
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
                
                processed += 1
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{len(candidates)}, found {len(final_candidates)} final candidates")
                
            except Exception:
                continue
        
        # Sort by explosive score
        final_candidates.sort(key=lambda x: x['explosive_score'], reverse=True)
        logger.info(f"✅ Final filtering complete: {len(final_candidates)} explosive candidates")
        
        return final_candidates
    
    def _calculate_explosive_score(self, stock_data: Dict[str, Any]) -> int:
        """Calculate explosive potential score"""
        score = 50.0
        
        # Short interest bonus (key for squeezes)
        score += min(stock_data['short_interest'] * 2, 40)
        
        # Volume surge bonus
        score += min((stock_data['volume_surge'] - 1) * 15, 25)
        
        # Small cap bonus
        if stock_data['market_cap'] < 1e9:
            score += 20
        elif stock_data['market_cap'] < 5e9:
            score += 10
        
        # Volatility energy bonus
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
    
    def optimized_scan(self) -> List[Dict[str, Any]]:
        """Complete optimized scan process"""
        logger.info("🚀 Starting OPTIMIZED universal scan...")
        
        # Step 1: Pre-filter entire universe using Polygon data only
        pre_filtered = self.get_entire_universe_with_prefiltering()
        
        if not pre_filtered:
            logger.error("No candidates from pre-filtering!")
            return []
        
        # Step 2: Apply detailed filters only to pre-filtered candidates
        final_candidates = self.apply_detailed_filters(pre_filtered)
        
        logger.info("📊 OPTIMIZATION SUMMARY:")
        logger.info(f"   Original Universe: ~5000+ stocks")
        logger.info(f"   Pre-filtered: {len(pre_filtered)} candidates")
        logger.info(f"   Final explosive candidates: {len(final_candidates)}")
        logger.info(f"   API call reduction: {((5000 - len(pre_filtered)) / 5000) * 100:.1f}%")
        
        return final_candidates

def main():
    """Test the optimized scanner"""
    scanner = OptimizedUniversalScanner()
    
    # Run optimized scan
    results = scanner.optimized_scan()
    
    print("\n🚀 OPTIMIZED UNIVERSAL SCAN RESULTS")
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