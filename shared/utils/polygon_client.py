#!/usr/bin/env python3
"""
Polygon API Client for Real-Time Stock Data
Provides explosive growth detection capabilities using Polygon.io API
"""

import requests
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class ExplosiveGrowthSignal:
    symbol: str
    current_price: float
    price_change_percent: float
    volume_surge_percent: float
    market_cap: Optional[float]
    signal_strength: float  # 0-1 scale
    detection_time: datetime
    triggers: List[str]  # What triggered this as explosive growth
    risk_score: float  # 0-1 scale, higher = more risky

class PolygonClient:
    """Enhanced Polygon API client for explosive growth detection"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        
        # Explosive growth detection parameters
        self.growth_thresholds = {
            'min_price_surge': 5.0,      # Minimum 5% price increase
            'min_volume_surge': 150.0,    # Minimum 150% volume increase
            'max_price': 500.0,          # Focus on stocks under $500
            'min_market_cap': 100e6,     # Minimum $100M market cap
            'max_market_cap': 50e9,      # Maximum $50B market cap (exclude mega caps)
            'lookback_minutes': 30,      # Look at last 30 minutes
            'min_avg_volume': 100000,    # Minimum average daily volume
        }
        
    async def get_explosive_growth_candidates(self, limit: int = 50) -> List[ExplosiveGrowthSignal]:
        """
        Find stocks showing explosive growth patterns in real-time
        """
        try:
            # Get real-time market data
            gainers = await self._get_market_gainers(limit=100)
            volume_leaders = await self._get_volume_leaders(limit=100)
            
            # Cross-reference and analyze
            explosive_candidates = []
            
            for stock_data in gainers + volume_leaders:
                signal = await self._analyze_explosive_potential(stock_data)
                if signal and signal.signal_strength > 0.6:  # High confidence threshold
                    explosive_candidates.append(signal)
            
            # Sort by signal strength and return top candidates
            explosive_candidates.sort(key=lambda x: x.signal_strength, reverse=True)
            return explosive_candidates[:limit]
            
        except Exception as e:
            logger.error(f"Error finding explosive growth candidates: {e}")
            return []
    
    async def _get_market_gainers(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get today's top gaining stocks"""
        try:
            url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/gainers"
            params = {
                'apikey': self.api_key,
                'include_otc': 'false'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('results', [])[:limit]
                    else:
                        logger.error(f"Error getting gainers: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error in _get_market_gainers: {e}")
            return []
    
    async def _get_volume_leaders(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get stocks with unusual volume activity"""
        try:
            url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers"
            params = {
                'apikey': self.api_key,
                'sort': 'volume',
                'order': 'desc',
                'limit': limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('results', [])
                    else:
                        logger.error(f"Error getting volume leaders: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error in _get_volume_leaders: {e}")
            return []
    
    async def _analyze_explosive_potential(self, stock_data: Dict[str, Any]) -> Optional[ExplosiveGrowthSignal]:
        """
        Analyze if a stock shows explosive growth potential
        """
        try:
            ticker = stock_data.get('ticker', '')
            if not ticker:
                return None
            
            # Get detailed data
            current_price = stock_data.get('value', 0)
            day_change_pct = stock_data.get('todaysChangePerc', 0)
            volume = stock_data.get('volume', 0)
            
            # Get additional metrics
            market_cap = await self._get_market_cap(ticker)
            avg_volume = await self._get_average_volume(ticker)
            
            # Apply explosive growth criteria
            triggers = []
            signal_strength = 0.0
            
            # Price surge check
            if day_change_pct >= self.growth_thresholds['min_price_surge']:
                triggers.append(f"Price surge: +{day_change_pct:.1f}%")
                signal_strength += 0.3
                
                # Bonus for bigger surges
                if day_change_pct >= 10:
                    signal_strength += 0.2
                if day_change_pct >= 20:
                    signal_strength += 0.2
            
            # Volume surge check
            if avg_volume > 0:
                volume_surge_pct = ((volume - avg_volume) / avg_volume) * 100
                if volume_surge_pct >= self.growth_thresholds['min_volume_surge']:
                    triggers.append(f"Volume surge: +{volume_surge_pct:.0f}%")
                    signal_strength += 0.3
            else:
                volume_surge_pct = 0
            
            # Market cap filtering
            if market_cap:
                if (self.growth_thresholds['min_market_cap'] <= market_cap <= 
                    self.growth_thresholds['max_market_cap']):
                    signal_strength += 0.1
                else:
                    signal_strength -= 0.2  # Penalize if outside optimal range
            
            # Price filtering  
            if current_price <= self.growth_thresholds['max_price']:
                signal_strength += 0.1
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(current_price, day_change_pct, 
                                                  volume_surge_pct, market_cap)
            
            # Only return if we have at least one strong trigger
            if len(triggers) > 0 and signal_strength > 0.5:
                return ExplosiveGrowthSignal(
                    symbol=ticker,
                    current_price=current_price,
                    price_change_percent=day_change_pct,
                    volume_surge_percent=volume_surge_pct,
                    market_cap=market_cap,
                    signal_strength=min(signal_strength, 1.0),
                    detection_time=datetime.now(),
                    triggers=triggers,
                    risk_score=risk_score
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing explosive potential for {stock_data}: {e}")
            return None
    
    async def _get_market_cap(self, ticker: str) -> Optional[float]:
        """Get market cap for a ticker"""
        try:
            url = f"{self.base_url}/v3/reference/tickers/{ticker}"
            params = {'apikey': self.api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('results', {}).get('market_cap')
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting market cap for {ticker}: {e}")
            return None
    
    async def _get_average_volume(self, ticker: str, days: int = 20) -> float:
        """Get average volume over specified days"""
        try:
            # Get historical data for volume calculation
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
            params = {'apikey': self.api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', [])
                        if results:
                            volumes = [r.get('v', 0) for r in results]
                            return sum(volumes) / len(volumes)
                    return 0
                    
        except Exception as e:
            logger.error(f"Error getting average volume for {ticker}: {e}")
            return 0
    
    def _calculate_risk_score(self, price: float, price_change_pct: float, 
                            volume_surge_pct: float, market_cap: Optional[float]) -> float:
        """
        Calculate risk score (0-1, higher = more risky)
        """
        risk_score = 0.0
        
        # High price change = higher risk
        if price_change_pct > 20:
            risk_score += 0.3
        elif price_change_pct > 10:
            risk_score += 0.2
        
        # Very high volume surge can indicate pump & dump
        if volume_surge_pct > 500:
            risk_score += 0.3
        
        # Very small market caps are riskier
        if market_cap and market_cap < 500e6:  # Under $500M
            risk_score += 0.2
        
        # Very high prices can be more volatile
        if price > 200:
            risk_score += 0.1
        
        return min(risk_score, 1.0)
    
    async def get_real_time_quote(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote for a specific ticker"""
        try:
            url = f"{self.base_url}/v2/last/trade/{ticker}"
            params = {'apikey': self.api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('results')
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting real-time quote for {ticker}: {e}")
            return None
    
    async def get_stock_universe(self, min_market_cap: float = 100e6) -> List[str]:
        """
        Get expanded stock universe from Polygon
        Returns list of tickers that meet minimum criteria
        """
        try:
            url = f"{self.base_url}/v3/reference/tickers"
            params = {
                'apikey': self.api_key,
                'market': 'stocks',
                'exchange': 'XNAS,XNYS',  # NASDAQ and NYSE
                'active': 'true',
                'limit': 1000
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        tickers = []
                        
                        for result in data.get('results', []):
                            market_cap = result.get('market_cap', 0)
                            if market_cap and market_cap >= min_market_cap:
                                tickers.append(result.get('ticker'))
                        
                        return [t for t in tickers if t]  # Filter out None values
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting stock universe: {e}")
            return []