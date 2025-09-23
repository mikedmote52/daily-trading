#!/usr/bin/env python3
"""
MCP Integration Example for Daily Trading System
Shows how to connect and use Polygon MCP for enhanced data
"""

import os
import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Method 1: Direct MCP Functions (when running in Claude Code)
def use_mcp_direct():
    """Use MCP functions directly when available in Claude Code environment"""
    try:
        # These functions are available when running in Claude Code
        # They're injected into the global namespace

        # Get short interest data
        short_data = mcp__polygon__list_short_interest(
            ticker="AAPL",
            limit=5
        )

        # Get market snapshot for all tickers
        snapshot = mcp__polygon__get_snapshot_all(
            market_type="stocks"
        )

        # Get Benzinga news sentiment
        news = mcp__polygon__list_benzinga_news(
            tickers="AAPL,TSLA",
            limit=10
        )

        # Get analyst insights
        analyst_data = mcp__polygon__list_benzinga_analyst_insights(
            ticker="AAPL",
            limit=5
        )

        return {
            "short_interest": short_data,
            "snapshot": snapshot,
            "news": news,
            "analyst": analyst_data
        }
    except NameError:
        print("MCP functions not available - use HTTP method")
        return None


# Method 2: HTTP MCP Server (for Render deployment)
class PolygonMCPClient:
    """Client for connecting to Polygon MCP server on Render"""

    def __init__(self):
        self.server_url = os.getenv('MCP_POLYGON_URL', 'https://polygon-mcp-server.onrender.com/mcp')
        self.api_key = os.getenv('POLYGON_API_KEY', '1ORwpSzeOV20X6uaA8G3Zuxx7hLJ0KIC')

    async def call_function(self, function_name: str, **params) -> Dict:
        """Call MCP function via HTTP"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": function_name,
                "params": params
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            async with session.post(
                self.server_url,
                json=payload,
                headers=headers
            ) as response:
                result = await response.json()
                return result.get('result', result)

    async def get_short_interest(self, ticker: str) -> Dict:
        """Get short interest data for a ticker"""
        return await self.call_function(
            'mcp__polygon__list_short_interest',
            ticker=ticker,
            limit=1
        )

    async def get_options_flow(self, ticker: str) -> Dict:
        """Get options activity via snapshot"""
        return await self.call_function(
            'mcp__polygon__get_snapshot_option',
            underlying_asset=ticker,
            option_contract=f"O:{ticker}*"  # Get all options for ticker
        )

    async def get_market_sentiment(self, tickers: List[str]) -> Dict:
        """Get market sentiment from news and analyst data"""
        # Get news sentiment
        news = await self.call_function(
            'mcp__polygon__list_benzinga_news',
            tickers=','.join(tickers),
            limit=20
        )

        # Get analyst insights
        analyst_data = {}
        for ticker in tickers[:5]:  # Limit to avoid rate limits
            analyst = await self.call_function(
                'mcp__polygon__list_benzinga_analyst_insights',
                ticker=ticker,
                limit=5
            )
            analyst_data[ticker] = analyst

        return {
            'news': news,
            'analyst': analyst_data
        }

    async def get_enhanced_data(self, ticker: str) -> Dict:
        """Get all enhanced data for a ticker"""
        tasks = [
            self.get_short_interest(ticker),
            self.get_options_flow(ticker),
            self.get_market_sentiment([ticker])
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            'short_interest': results[0] if not isinstance(results[0], Exception) else None,
            'options_flow': results[1] if not isinstance(results[1], Exception) else None,
            'sentiment': results[2] if not isinstance(results[2], Exception) else None
        }


# Method 3: Direct Polygon API (fallback)
def use_polygon_api_fallback():
    """Use Polygon REST API directly as fallback"""
    from polygon import RESTClient

    api_key = os.getenv('POLYGON_API_KEY', '1ORwpSzeOV20X6uaA8G3Zuxx7hLJ0KIC')
    client = RESTClient(api_key)

    # Get short interest
    short_data = client.list_short_interest(
        ticker="AAPL",
        limit=1
    )

    # Get market snapshot
    snapshot = client.get_snapshot_all('stocks')

    return {
        'short_interest': list(short_data),
        'snapshot': snapshot
    }


# Integration with Discovery System
class EnhancedDiscoveryData:
    """Enhanced data provider for discovery system"""

    def __init__(self):
        self.mcp_client = PolygonMCPClient()
        self.cache = {}
        self.cache_duration = 3600  # 1 hour

    async def enrich_discovery_candidates(self, tickers: List[str]) -> Dict[str, Dict]:
        """Enrich discovery candidates with MCP data"""
        enhanced_data = {}

        for ticker in tickers:
            # Check cache
            cache_key = f"{ticker}_enhanced"
            if cache_key in self.cache:
                cache_age = datetime.now().timestamp() - self.cache[cache_key]['timestamp']
                if cache_age < self.cache_duration:
                    enhanced_data[ticker] = self.cache[cache_key]['data']
                    continue

            # Fetch fresh data
            try:
                data = await self.mcp_client.get_enhanced_data(ticker)

                # Process and score the data
                score_adjustments = self._calculate_score_adjustments(data)

                enhanced_data[ticker] = {
                    'raw_data': data,
                    'score_adjustments': score_adjustments,
                    'timestamp': datetime.now().isoformat()
                }

                # Cache it
                self.cache[cache_key] = {
                    'data': enhanced_data[ticker],
                    'timestamp': datetime.now().timestamp()
                }

            except Exception as e:
                print(f"Error enriching {ticker}: {e}")
                enhanced_data[ticker] = None

        return enhanced_data

    def _calculate_score_adjustments(self, data: Dict) -> Dict:
        """Calculate scoring adjustments based on enhanced data"""
        adjustments = {
            'short_squeeze_potential': 0,
            'options_momentum': 0,
            'sentiment_score': 0
        }

        # Short squeeze potential
        if data.get('short_interest'):
            try:
                si_data = data['short_interest'].get('results', [{}])[0]
                days_to_cover = si_data.get('days_to_cover', 0)
                if days_to_cover > 3:
                    adjustments['short_squeeze_potential'] = min(20, days_to_cover * 2)
            except:
                pass

        # Options momentum (simplified)
        if data.get('options_flow'):
            # Look for unusual options activity
            adjustments['options_momentum'] = 10  # Placeholder

        # Sentiment scoring
        if data.get('sentiment', {}).get('news'):
            try:
                news_items = data['sentiment']['news'].get('results', [])
                positive_count = sum(1 for item in news_items
                                   if 'bullish' in str(item).lower() or
                                      'upgrade' in str(item).lower())
                adjustments['sentiment_score'] = min(15, positive_count * 3)
            except:
                pass

        return adjustments


# Example usage
async def main():
    """Example of using MCP in discovery system"""

    # Initialize enhanced data provider
    provider = EnhancedDiscoveryData()

    # Example tickers from discovery
    discovery_candidates = ["AAPL", "TSLA", "GME", "AMC"]

    print("Enriching discovery candidates with MCP data...")
    enhanced_data = await provider.enrich_discovery_candidates(discovery_candidates)

    # Display results
    for ticker, data in enhanced_data.items():
        if data:
            print(f"\n{ticker}:")
            adjustments = data['score_adjustments']
            print(f"  Short Squeeze Potential: +{adjustments['short_squeeze_potential']}")
            print(f"  Options Momentum: +{adjustments['options_momentum']}")
            print(f"  Sentiment Score: +{adjustments['sentiment_score']}")
            total_boost = sum(adjustments.values())
            print(f"  Total Score Boost: +{total_boost}")


if __name__ == "__main__":
    # Check which method is available
    print("Testing MCP availability...")

    # Try direct MCP first
    direct_result = use_mcp_direct()
    if direct_result:
        print("✅ Direct MCP functions available")
        print(json.dumps(direct_result, indent=2)[:500])
    else:
        print("⚠️  Direct MCP not available, using HTTP method")

        # Use async HTTP method
        asyncio.run(main())