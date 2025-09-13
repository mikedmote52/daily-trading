#!/usr/bin/env python3
"""
Universal Stock Scanner - Production Version

Scans the ENTIRE universe of stocks with systematic filtering.
Ready for GitHub/Render deployment and UI integration.
"""

import asyncio
import logging
import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('UniversalScanner')

@dataclass
class ScanFilters:
    """Production-ready filtering criteria"""
    max_price: float = 100.0           # Hard cap at $100
    min_price: float = 1.0             # No penny stocks under $1
    min_volume: int = 500000           # 500K minimum daily volume
    min_market_cap: float = 100e6      # $100M minimum
    max_market_cap: float = 50e9       # $50B maximum
    min_short_interest: float = 5.0    # 5% minimum short interest
    min_volatility: float = 0.25       # 25% annual volatility minimum
    max_volatility: float = 2.0        # 200% maximum (avoid extreme)
    volume_surge_min: float = 1.5      # 1.5x volume surge minimum
    max_pe_ratio: float = 50.0         # Growth-friendly P/E limit

@dataclass
class StockResult:
    symbol: str
    price: float
    market_cap: float
    volume: int
    volume_surge: float
    volatility: float
    short_interest: float
    pe_ratio: Optional[float]
    momentum_score: float
    explosive_score: int
    sector: str
    exchange: str

class UniversalStockScanner:
    def __init__(self):
        self.filters = ScanFilters()
        self.results: List[StockResult] = []
        
    def get_all_stocks_polygon(self) -> List[str]:
        """Get all stocks using Polygon API"""
        try:
            api_key = "1ORwpSzeOV20X6uaA8G3Zuxx7hLJ0KIC"
            url = "https://api.polygon.io/v3/reference/tickers"
            
            all_stocks = []
            next_url = url
            
            while next_url:
                params = {
                    'apikey': api_key,
                    'market': 'stocks',
                    'active': 'true',
                    'limit': 1000  # Maximum per request
                }
                
                # If continuing pagination, use the next_url
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
                            
                            # Only include common stocks (CS), filter out ETFs, ADRs, etc.
                            if (symbol and 
                                ticker_type == 'CS' and
                                len(symbol) <= 5 and 
                                symbol.isalpha() and
                                not any(char in symbol for char in ['.', '-', '+'])):
                                all_stocks.append(symbol)
                    
                    # Check for pagination
                    next_url = data.get('next_url')
                    if next_url:
                        next_url += f"&apikey={api_key}"
                    
                    logger.info(f"Retrieved {len(all_stocks)} stocks so far...")
                    
                    # Rate limiting for Polygon API
                    time.sleep(0.1)
                    
                else:
                    logger.error(f"Polygon API error: {response.status_code}")
                    break
            
            logger.info(f"Total stocks retrieved from Polygon: {len(all_stocks)}")
            return all_stocks
            
        except Exception as e:
            logger.error(f"Failed to fetch stocks from Polygon: {e}")
            return []
    
    def get_entire_stock_universe(self) -> List[str]:
        """Get the complete universe of US stocks using Polygon API"""
        logger.info("🌍 Fetching ENTIRE stock universe from Polygon...")
        
        # Get all stocks from Polygon
        all_stocks = self.get_all_stocks_polygon()
        
        # Additional filtering for production readiness
        filtered_stocks = []
        for symbol in all_stocks:
            # Strict filtering for explosive growth candidates
            if (len(symbol) <= 5 and 
                symbol.isalpha() and
                not any(keyword in symbol.upper() for keyword in 
                       ['TEST', 'TEMP', 'OLD', 'NEW']) and
                len(symbol) >= 2):  # Minimum 2 characters
                filtered_stocks.append(symbol)
        
        # Remove duplicates and sort
        unique_stocks = sorted(list(set(filtered_stocks)))
        
        logger.info(f"📊 Total stock universe: {len(unique_stocks)} stocks")
        return unique_stocks
    
    async def apply_immediate_filters(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Apply immediate filters - price, volume, market cap, etc."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get basic info first for quick filtering
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price:
                return None
            
            # IMMEDIATE FILTER 1: Price must be under $100
            if current_price > self.filters.max_price or current_price < self.filters.min_price:
                return None
            
            # IMMEDIATE FILTER 2: Market cap range
            market_cap = info.get('marketCap', 0)
            if (market_cap < self.filters.min_market_cap or 
                market_cap > self.filters.max_market_cap):
                return None
            
            # IMMEDIATE FILTER 3: Must be a stock, not fund/ETF
            security_type = info.get('quoteType', '').upper()
            if security_type in ['ETF', 'MUTUALFUND', 'INDEX']:
                return None
            
            # Get historical data for volume/volatility analysis
            hist = ticker.history(period="1mo")
            if len(hist) < 20:  # Need sufficient data
                return None
            
            # IMMEDIATE FILTER 4: Volume requirements
            current_volume = hist['Volume'].iloc[-1]
            if current_volume < self.filters.min_volume:
                return None
            
            # IMMEDIATE FILTER 5: Volume surge requirement
            avg_volume_20d = hist['Volume'].rolling(20).mean().iloc[-1]
            volume_surge = current_volume / avg_volume_20d if avg_volume_20d > 0 else 0
            if volume_surge < self.filters.volume_surge_min:
                return None
            
            # IMMEDIATE FILTER 6: Volatility requirements
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            if (volatility < self.filters.min_volatility or 
                volatility > self.filters.max_volatility):
                return None
            
            # IMMEDIATE FILTER 7: Short interest requirement
            short_interest = info.get('shortPercentOfFloat', 0) * 100
            if short_interest < self.filters.min_short_interest:
                return None
            
            # IMMEDIATE FILTER 8: P/E ratio filter
            pe_ratio = info.get('trailingPE')
            if pe_ratio and pe_ratio > self.filters.max_pe_ratio:
                return None
            
            # If we get here, stock passed all immediate filters
            return {
                'symbol': symbol,
                'price': current_price,
                'market_cap': market_cap,
                'volume': int(current_volume),
                'volume_surge': volume_surge,
                'volatility': volatility,
                'short_interest': short_interest,
                'pe_ratio': pe_ratio,
                'sector': info.get('sector', 'Unknown'),
                'exchange': info.get('exchange', 'Unknown'),
                'history': hist,
                'info': info
            }
            
        except Exception as e:
            # Silently skip problematic stocks
            return None
    
    def calculate_explosive_score(self, stock_data: Dict[str, Any]) -> int:
        """Calculate explosive potential score (0-100)"""
        score = 50.0  # Base score
        
        # Short interest bonus (key for squeezes)
        si_bonus = min(stock_data['short_interest'] * 2, 40)
        score += si_bonus
        
        # Volume surge bonus
        volume_bonus = min((stock_data['volume_surge'] - 1) * 15, 25)
        score += volume_bonus
        
        # Small cap bonus (smaller = more explosive potential)
        market_cap = stock_data['market_cap']
        if market_cap < 1e9:  # < $1B
            score += 20
        elif market_cap < 5e9:  # < $5B
            score += 10
        
        # Volatility energy bonus
        vol_bonus = min(stock_data['volatility'] * 15, 20)
        score += vol_bonus
        
        # Momentum calculation
        hist = stock_data['history']
        if len(hist) >= 20:
            momentum = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-20]) - 1) * 100
            momentum_bonus = min(abs(momentum) * 0.5, 15)
            score += momentum_bonus
        
        # Sector bonus for explosive sectors
        sector_bonuses = {
            'Healthcare': 10,
            'Technology': 8,
            'Communication Services': 6,
            'Consumer Discretionary': 5,
            'Energy': 7
        }
        sector_bonus = sector_bonuses.get(stock_data['sector'], 2)
        score += sector_bonus
        
        return int(min(max(score, 0), 100))
    
    async def scan_batch(self, symbols: List[str]) -> List[StockResult]:
        """Scan a batch of symbols with immediate filtering"""
        results = []
        
        for symbol in symbols:
            try:
                # Apply immediate filters
                stock_data = await self.apply_immediate_filters(symbol)
                
                if stock_data:  # Passed all filters
                    # Calculate explosive score
                    explosive_score = self.calculate_explosive_score(stock_data)
                    
                    # Calculate momentum
                    hist = stock_data['history']
                    momentum = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-20]) - 1) * 100 if len(hist) >= 20 else 0
                    
                    # Create result
                    result = StockResult(
                        symbol=symbol,
                        price=stock_data['price'],
                        market_cap=stock_data['market_cap'],
                        volume=stock_data['volume'],
                        volume_surge=stock_data['volume_surge'],
                        volatility=stock_data['volatility'],
                        short_interest=stock_data['short_interest'],
                        pe_ratio=stock_data['pe_ratio'],
                        momentum_score=momentum,
                        explosive_score=explosive_score,
                        sector=stock_data['sector'],
                        exchange=stock_data['exchange']
                    )
                    
                    results.append(result)
                    logger.info(f"✅ {symbol}: Score {explosive_score}, Price ${stock_data['price']:.2f}")
                
            except Exception as e:
                # Skip problematic stocks silently
                continue
        
        return results
    
    async def scan_entire_universe(self, batch_size: int = 50, max_results: int = 100) -> List[StockResult]:
        """Scan the entire stock universe with systematic filtering"""
        logger.info("🚀 Starting UNIVERSAL stock scan...")
        
        # Get entire stock universe
        all_stocks = self.get_entire_stock_universe()
        logger.info(f"📊 Scanning {len(all_stocks)} stocks from entire universe")
        
        all_results = []
        processed = 0
        
        # Process in batches to avoid rate limits
        for i in range(0, len(all_stocks), batch_size):
            batch = all_stocks[i:i + batch_size]
            
            logger.info(f"🔍 Processing batch {i//batch_size + 1}: symbols {i+1}-{min(i+batch_size, len(all_stocks))}")
            
            # Scan batch
            batch_results = await self.scan_batch(batch)
            all_results.extend(batch_results)
            
            processed += len(batch)
            logger.info(f"📈 Progress: {processed}/{len(all_stocks)} ({processed/len(all_stocks)*100:.1f}%) - Found {len(all_results)} candidates")
            
            # Stop if we have enough high-quality results
            if len(all_results) >= max_results:
                logger.info(f"🎯 Reached target of {max_results} candidates")
                break
            
            # Rate limiting
            await asyncio.sleep(1)
        
        # Sort by explosive score
        all_results.sort(key=lambda x: x.explosive_score, reverse=True)
        
        logger.info(f"✅ Universal scan complete: {len(all_results)} explosive candidates found")
        return all_results
    
    def generate_scan_report(self, results: List[StockResult]) -> Dict[str, Any]:
        """Generate comprehensive scan report for UI integration"""
        if not results:
            return {"error": "No candidates found"}
        
        # Top candidates
        top_10 = results[:10]
        
        # Statistics
        avg_score = np.mean([r.explosive_score for r in results])
        avg_price = np.mean([r.price for r in results])
        total_market_cap = sum([r.market_cap for r in results])
        
        # Sector breakdown
        sector_counts = {}
        for result in results:
            sector_counts[result.sector] = sector_counts.get(result.sector, 0) + 1
        
        # Exchange breakdown
        exchange_counts = {}
        for result in results:
            exchange_counts[result.exchange] = exchange_counts.get(result.exchange, 0) + 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_candidates": len(results),
            "scan_filters": asdict(self.filters),
            "statistics": {
                "average_explosive_score": round(avg_score, 1),
                "average_price": round(avg_price, 2),
                "total_market_cap": total_market_cap,
                "high_score_count": len([r for r in results if r.explosive_score >= 80]),
                "max_score": max([r.explosive_score for r in results]),
                "min_score": min([r.explosive_score for r in results])
            },
            "sector_breakdown": dict(sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)),
            "exchange_breakdown": exchange_counts,
            "top_candidates": [
                {
                    "rank": i + 1,
                    "symbol": result.symbol,
                    "price": result.price,
                    "explosive_score": result.explosive_score,
                    "market_cap_billions": round(result.market_cap / 1e9, 2),
                    "volume_surge": round(result.volume_surge, 1),
                    "short_interest": round(result.short_interest, 1),
                    "momentum": round(result.momentum_score, 1),
                    "sector": result.sector,
                    "exchange": result.exchange
                }
                for i, result in enumerate(top_10)
            ]
        }

async def main():
    """Production test of universal scanner"""
    scanner = UniversalStockScanner()
    
    # Run universal scan
    results = await scanner.scan_entire_universe(batch_size=25, max_results=50)
    
    # Generate report
    report = scanner.generate_scan_report(results)
    
    # Save results for API/UI integration
    with open('universal_scan_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Display top results
    print("\n🚀 UNIVERSAL STOCK SCAN RESULTS")
    print("=" * 60)
    print(f"📊 Total Candidates Found: {report['total_candidates']}")
    print(f"📈 Average Explosive Score: {report['statistics']['average_explosive_score']}")
    print(f"💰 Average Price: ${report['statistics']['average_price']}")
    
    print("\n🏆 TOP 10 EXPLOSIVE CANDIDATES:")
    for candidate in report['top_candidates']:
        print(f"{candidate['rank']:2d}. {candidate['symbol']:6s} - ${candidate['price']:7.2f} - Score: {candidate['explosive_score']:3d}/100")
        print(f"    💰 MCap: ${candidate['market_cap_billions']:4.1f}B | 🔊 Vol: {candidate['volume_surge']:4.1f}x | ⚡ SI: {candidate['short_interest']:4.1f}%")
    
    print(f"\n💾 Results saved to: universal_scan_results.json")

if __name__ == "__main__":
    asyncio.run(main())