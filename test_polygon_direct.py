#!/usr/bin/env python3
"""
Direct test of Polygon API to verify connectivity and data
"""

import requests
import json
from datetime import datetime

def test_polygon_direct():
    """Test Polygon API directly"""
    api_key = "1ORwpSzeOV20X6uaA8G3Zuxx7hLJ0KIC"
    base_url = "https://api.polygon.io"
    
    print("🔍 Direct Polygon API Test")
    print(f"⏰ Test Time: {datetime.now()}")
    print("="*50)
    
    # Test 1: Basic API connectivity
    print("\n📡 Testing API Connectivity...")
    try:
        url = f"{base_url}/v2/snapshot/locale/us/markets/stocks/tickers"
        params = {
            'apikey': api_key,
            'limit': 5
        }
        
        response = requests.get(url, params=params, timeout=10)
        print(f"  Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ API Connected Successfully!")
            print(f"  📊 Response Keys: {list(data.keys())}")
            
            if 'results' in data:
                results = data['results']
                print(f"  📈 Found {len(results)} stock snapshots")
                
                for i, stock in enumerate(results[:3]):
                    ticker = stock.get('ticker', 'N/A')
                    value = stock.get('value', 0)
                    change = stock.get('todaysChangePerc', 0)
                    volume = stock.get('volume', 0)
                    
                    print(f"    {i+1}. {ticker}: ${value:.2f} ({change:+.1f}%) Vol: {volume:,}")
            else:
                print(f"  ⚠️ No results in response: {data}")
        else:
            print(f"  ❌ API Error: {response.status_code}")
            print(f"  Response: {response.text}")
            
    except Exception as e:
        print(f"  ❌ Connection Error: {e}")
    
    # Test 2: Market status
    print("\n🕐 Testing Market Status...")
    try:
        url = f"{base_url}/v1/marketstatus/now"
        params = {'apikey': api_key}
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Market Status Retrieved")
            
            if 'market' in data:
                market_status = data['market']
                print(f"  📈 Market: {market_status}")
            
            if 'exchanges' in data:
                exchanges = data['exchanges']
                print(f"  🏢 Exchanges:")
                for exchange_name, exchange_info in exchanges.items():
                    status = exchange_info.get('market', 'unknown')
                    print(f"    {exchange_name}: {status}")
        else:
            print(f"  ❌ Market Status Error: {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Market Status Error: {e}")
    
    # Test 3: Get some specific stock data
    print("\n📊 Testing Specific Stock Data (AAPL)...")
    try:
        url = f"{base_url}/v2/snapshot/locale/us/markets/stocks/tickers/AAPL"
        params = {'apikey': api_key}
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ AAPL Data Retrieved")
            
            if 'results' in data and data['results']:
                aapl = data['results']
                print(f"  📈 AAPL Details:")
                print(f"    Price: ${aapl.get('value', 0):.2f}")
                print(f"    Change: {aapl.get('todaysChangePerc', 0):+.2f}%")
                print(f"    Volume: {aapl.get('volume', 0):,}")
        else:
            print(f"  ❌ AAPL Data Error: {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ AAPL Data Error: {e}")
    
    print("\n" + "="*50)
    print("🎯 Direct API Test Complete!")

if __name__ == "__main__":
    test_polygon_direct()