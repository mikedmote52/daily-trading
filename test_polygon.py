#!/usr/bin/env python3
"""
Test Polygon API connection and explosive growth detection
"""

import asyncio
import sys
import os
from datetime import datetime

# Add shared utilities to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))

try:
    from utils.polygon_client import PolygonClient
    
    async def test_polygon_api():
        """Test Polygon API functionality"""
        api_key = "1ORwpSzeOV20X6uaA8G3Zuxx7hLJ0KIC"
        client = PolygonClient(api_key)
        
        print("🔍 Testing Polygon API Connection...")
        print(f"⏰ Test Time: {datetime.now()}")
        print("="*50)
        
        # Test 1: Get market gainers
        print("\n📈 Testing Market Gainers...")
        try:
            gainers = await client._get_market_gainers(limit=5)
            if gainers:
                print(f"✅ Found {len(gainers)} gainers")
                for i, gainer in enumerate(gainers[:3]):
                    print(f"  {i+1}. {gainer.get('ticker', 'N/A')} - "
                          f"${gainer.get('value', 0):.2f} "
                          f"({gainer.get('todaysChangePerc', 0):.1f}%)")
            else:
                print("❌ No gainers found")
        except Exception as e:
            print(f"❌ Error getting gainers: {e}")
        
        # Test 2: Get explosive growth candidates
        print("\n🚀 Testing Explosive Growth Detection...")
        try:
            explosive_signals = await client.get_explosive_growth_candidates(limit=5)
            if explosive_signals:
                print(f"✅ Found {len(explosive_signals)} explosive growth candidates!")
                for signal in explosive_signals:
                    print(f"\n  🎯 {signal.symbol}:")
                    print(f"     Price: ${signal.current_price:.2f} ({signal.price_change_percent:+.1f}%)")
                    print(f"     Volume Surge: {signal.volume_surge_percent:.0f}%")
                    print(f"     Signal Strength: {signal.signal_strength:.2f}/1.0")
                    print(f"     Risk Score: {signal.risk_score:.2f}/1.0")
                    print(f"     Triggers: {', '.join(signal.triggers)}")
            else:
                print("⚠️ No explosive growth candidates found (market may be closed or low volatility)")
        except Exception as e:
            print(f"❌ Error detecting explosive growth: {e}")
        
        # Test 3: Get stock universe
        print("\n🌎 Testing Stock Universe...")
        try:
            universe = await client.get_stock_universe(min_market_cap=1e9)
            if universe:
                print(f"✅ Retrieved {len(universe)} stocks in universe")
                print(f"  Sample tickers: {', '.join(universe[:10])}")
            else:
                print("❌ Failed to retrieve stock universe")
        except Exception as e:
            print(f"❌ Error getting stock universe: {e}")
        
        print("\n" + "="*50)
        print("🎯 Polygon API Test Complete!")
        
        return len(explosive_signals) if 'explosive_signals' in locals() and explosive_signals else 0

    if __name__ == "__main__":
        print("🚀 EXPLOSIVE GROWTH DETECTION SYSTEM TEST")
        results = asyncio.run(test_polygon_api())
        print(f"\n📊 Test Results: {results} explosive opportunities detected")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure the shared utilities are properly configured")
except Exception as e:
    print(f"❌ Unexpected error: {e}")