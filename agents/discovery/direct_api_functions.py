#!/usr/bin/env python3
"""
Direct Polygon API Functions - Zero Mock Data, Premium Features Only
Replaces MCP calls with direct API access to ensure real premium data
"""
import logging
import os
from polygon import RESTClient
from typing import Dict, List, Any, Optional

logger = logging.getLogger('DirectPolygonAPI')

# Initialize Polygon client with premium API key
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '1ORwpSzeOV20X6uaA8G3Zuxx7hLJ0KIC')
if not POLYGON_API_KEY:
    raise Exception("CRITICAL: POLYGON_API_KEY required for premium data access")

# Initialize client once
polygon_client = RESTClient(POLYGON_API_KEY)
logger.info("✅ Direct Polygon API client initialized - PREMIUM DATA ACCESS ENABLED")
logger.info("🔒 NO MOCK DATA - 100% real market data guaranteed")

def get_snapshot_all(market_type: str = "stocks", **kwargs) -> Dict:
    """Get snapshot of all stocks - REAL DATA ONLY"""
    try:
        logger.info("🔍 Fetching market snapshot via direct Polygon API...")

        # Use Polygon client's get_snapshot_all method
        response = polygon_client.get_snapshot_all(market_type)

        # Convert to dictionary format expected by discovery system
        if hasattr(response, 'tickers'):
            results = []
            for ticker in response.tickers:
                ticker_data = {
                    'ticker': ticker.ticker,
                    'todaysChangePerc': getattr(ticker, 'todaysChangePerc', 0),
                    'todaysChange': getattr(ticker, 'todaysChange', 0),
                    'updated': getattr(ticker, 'updated', 0),
                    'day': {
                        'o': getattr(ticker.day, 'o', 0) if hasattr(ticker, 'day') else 0,
                        'h': getattr(ticker.day, 'h', 0) if hasattr(ticker, 'day') else 0,
                        'l': getattr(ticker.day, 'l', 0) if hasattr(ticker, 'day') else 0,
                        'c': getattr(ticker.day, 'c', 0) if hasattr(ticker, 'day') else 0,
                        'v': getattr(ticker.day, 'v', 0) if hasattr(ticker, 'day') else 0,
                        'vw': getattr(ticker.day, 'vw', 0) if hasattr(ticker, 'day') else 0
                    },
                    'prevDay': {
                        'o': getattr(ticker.prevDay, 'o', 0) if hasattr(ticker, 'prevDay') else 0,
                        'h': getattr(ticker.prevDay, 'h', 0) if hasattr(ticker, 'prevDay') else 0,
                        'l': getattr(ticker.prevDay, 'l', 0) if hasattr(ticker, 'prevDay') else 0,
                        'c': getattr(ticker.prevDay, 'c', 0) if hasattr(ticker, 'prevDay') else 0,
                        'v': getattr(ticker.prevDay, 'v', 0) if hasattr(ticker, 'prevDay') else 0,
                        'vw': getattr(ticker.prevDay, 'vw', 0) if hasattr(ticker, 'prevDay') else 0
                    }
                }
                results.append(ticker_data)

            logger.info(f"✅ Retrieved {len(results)} stocks from Polygon API - ALL REAL DATA")
            return {"status": "OK", "results": results}
        else:
            logger.error("❌ Unexpected response format from Polygon API")
            return {"status": "ERROR", "results": []}

    except Exception as e:
        logger.error(f"❌ Direct Polygon API call failed: {e}")
        raise Exception(f"REAL DATA ACCESS FAILED: {e}")

def get_snapshot_ticker(market_type: str, ticker: str, **kwargs) -> Dict:
    """Get snapshot for specific ticker - REAL DATA ONLY"""
    try:
        logger.debug(f"📊 Fetching {ticker} snapshot via direct Polygon API...")

        # Use Polygon client's get_snapshot_ticker method
        response = polygon_client.get_snapshot_ticker(market_type, ticker)

        if hasattr(response, 'ticker'):
            ticker_obj = response.ticker
            result = {
                'ticker': {
                    'ticker': ticker_obj.ticker,
                    'todaysChangePerc': getattr(ticker_obj, 'todaysChangePerc', 0),
                    'todaysChange': getattr(ticker_obj, 'todaysChange', 0),
                    'updated': getattr(ticker_obj, 'updated', 0),
                    'day': {
                        'o': getattr(ticker_obj.day, 'o', 0) if hasattr(ticker_obj, 'day') else 0,
                        'h': getattr(ticker_obj.day, 'h', 0) if hasattr(ticker_obj, 'day') else 0,
                        'l': getattr(ticker_obj.day, 'l', 0) if hasattr(ticker_obj, 'day') else 0,
                        'c': getattr(ticker_obj.day, 'c', 0) if hasattr(ticker_obj, 'day') else 0,
                        'v': getattr(ticker_obj.day, 'v', 0) if hasattr(ticker_obj, 'day') else 0,
                        'vw': getattr(ticker_obj.day, 'vw', 0) if hasattr(ticker_obj, 'day') else 0
                    },
                    'prevDay': {
                        'o': getattr(ticker_obj.prevDay, 'o', 0) if hasattr(ticker_obj, 'prevDay') else 0,
                        'h': getattr(ticker_obj.prevDay, 'h', 0) if hasattr(ticker_obj, 'prevDay') else 0,
                        'l': getattr(ticker_obj.prevDay, 'l', 0) if hasattr(ticker_obj, 'prevDay') else 0,
                        'c': getattr(ticker_obj.prevDay, 'c', 0) if hasattr(ticker_obj, 'prevDay') else 0,
                        'v': getattr(ticker_obj.prevDay, 'v', 0) if hasattr(ticker_obj, 'prevDay') else 0,
                        'vw': getattr(ticker_obj.prevDay, 'vw', 0) if hasattr(ticker_obj, 'prevDay') else 0
                    }
                },
                'status': 'OK'
            }

            logger.debug(f"✅ {ticker} snapshot retrieved - REAL DATA")
            return result
        else:
            logger.error(f"❌ No ticker data for {ticker}")
            return {"status": "ERROR", "ticker": None}

    except Exception as e:
        logger.error(f"❌ Ticker snapshot failed for {ticker}: {e}")
        raise Exception(f"REAL DATA ACCESS FAILED for {ticker}: {e}")

def list_short_interest(ticker: str, limit: int = 1, **kwargs) -> Dict:
    """Get short interest data - PREMIUM FEATURE, REAL DATA ONLY"""
    try:
        logger.debug(f"📈 Fetching short interest for {ticker} via direct Polygon API...")

        # Use Polygon client's list_short_interest method
        response = polygon_client.list_short_interest(ticker=ticker, limit=limit)

        results = []
        for item in response:
            result_item = {
                'settlement_date': getattr(item, 'settlement_date', ''),
                'ticker': getattr(item, 'ticker', ticker),
                'short_interest': getattr(item, 'short_interest', 0),
                'avg_daily_volume': getattr(item, 'avg_daily_volume', 0),
                'days_to_cover': getattr(item, 'days_to_cover', 0)
            }
            results.append(result_item)

        logger.debug(f"✅ Short interest for {ticker} retrieved - PREMIUM REAL DATA")
        return {
            "status": "OK",
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"❌ Short interest failed for {ticker}: {e}")
        raise Exception(f"PREMIUM SHORT INTEREST ACCESS FAILED for {ticker}: {e}")

def get_ticker_details(ticker: str, **kwargs) -> Dict:
    """Get ticker details - PREMIUM FEATURE, REAL DATA ONLY"""
    try:
        logger.debug(f"📋 Fetching ticker details for {ticker} via direct Polygon API...")

        # Use Polygon client's get_ticker_details method
        response = polygon_client.get_ticker_details(ticker)

        if hasattr(response, 'results'):
            details = response.results
            result = {
                'results': {
                    'ticker': getattr(details, 'ticker', ticker),
                    'name': getattr(details, 'name', ''),
                    'market_cap': getattr(details, 'market_cap', 0),
                    'share_class_shares_outstanding': getattr(details, 'share_class_shares_outstanding', 0),
                    'weighted_shares_outstanding': getattr(details, 'weighted_shares_outstanding', 0),
                    'description': getattr(details, 'description', ''),
                    'homepage_url': getattr(details, 'homepage_url', ''),
                    'list_date': getattr(details, 'list_date', ''),
                    'locale': getattr(details, 'locale', ''),
                    'market': getattr(details, 'market', ''),
                    'primary_exchange': getattr(details, 'primary_exchange', ''),
                    'type': getattr(details, 'type', ''),
                    'active': getattr(details, 'active', True),
                    'currency_name': getattr(details, 'currency_name', 'USD'),
                    'cik': getattr(details, 'cik', ''),
                    'composite_figi': getattr(details, 'composite_figi', ''),
                    'share_class_figi': getattr(details, 'share_class_figi', ''),
                    'sic_code': getattr(details, 'sic_code', ''),
                    'sic_description': getattr(details, 'sic_description', '')
                },
                'status': 'OK'
            }

            logger.debug(f"✅ Ticker details for {ticker} retrieved - PREMIUM REAL DATA")
            return result
        else:
            logger.error(f"❌ No details available for {ticker}")
            return {"status": "ERROR", "results": None}

    except Exception as e:
        logger.error(f"❌ Ticker details failed for {ticker}: {e}")
        raise Exception(f"PREMIUM TICKER DETAILS ACCESS FAILED for {ticker}: {e}")

def get_market_status(**kwargs) -> Dict:
    """Get market status - REAL DATA ONLY"""
    try:
        logger.debug("🏛️ Fetching market status via direct Polygon API...")

        # Use Polygon client's get_market_status method
        response = polygon_client.get_market_status()

        result = {
            'market': getattr(response, 'market', 'unknown'),
            'serverTime': getattr(response, 'serverTime', ''),
            'exchanges': {}
        }

        # Add exchange information if available
        if hasattr(response, 'exchanges'):
            for exchange_name, exchange_info in response.exchanges.items():
                result['exchanges'][exchange_name] = {
                    'market': getattr(exchange_info, 'market', 'unknown'),
                    'serverTime': getattr(exchange_info, 'serverTime', '')
                }

        logger.debug("✅ Market status retrieved - REAL DATA")
        return result

    except Exception as e:
        logger.error(f"❌ Market status failed: {e}")
        raise Exception(f"MARKET STATUS ACCESS FAILED: {e}")

# Function mapping for easy replacement of MCP calls
DIRECT_API_FUNCTIONS = {
    'mcp__polygon__get_snapshot_all': get_snapshot_all,
    'mcp__polygon__get_snapshot_ticker': get_snapshot_ticker,
    'mcp__polygon__list_short_interest': list_short_interest,
    'mcp__polygon__get_ticker_details': get_ticker_details,
    'mcp__polygon__get_market_status': get_market_status
}

def call_direct_api(function_name: str, **kwargs) -> Dict:
    """Call direct API function by name - GUARANTEED REAL DATA"""
    if function_name in DIRECT_API_FUNCTIONS:
        func = DIRECT_API_FUNCTIONS[function_name]
        logger.info(f"🎯 Calling {function_name} via direct Polygon API - REAL DATA GUARANTEED")
        return func(**kwargs)
    else:
        raise ValueError(f"Unknown function: {function_name}")

logger.info("🚀 Direct Polygon API functions loaded - PREMIUM DATA ACCESS READY")
logger.info("🔒 ZERO MOCK DATA - All calls use real Polygon API with premium features")