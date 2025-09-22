#!/usr/bin/env python3
"""
MCP-powered discovery system that runs in Claude Code environment
This version uses MCP functions directly instead of HTTP fallbacks
"""

def run_mcp_discovery():
    """Run discovery using MCP functions available in Claude Code"""
    results = []

    # Test with a few known active stocks
    test_symbols = ['AAPL', 'TSLA', 'NVDA', 'AMZN', 'GOOGL', 'MSFT', 'META', 'AMD']

    print("🚀 RUNNING MCP-POWERED DISCOVERY")
    print("="*50)

    for symbol in test_symbols:
        try:
            # Get real-time snapshot data using MCP
            snapshot = mcp__polygon__get_snapshot_ticker(market_type="stocks", ticker=symbol)

            if snapshot and 'ticker' in snapshot:
                ticker_data = snapshot['ticker']

                # Extract key metrics
                price = ticker_data.get('prevDay', {}).get('c', 0)
                volume = ticker_data.get('prevDay', {}).get('v', 0)
                change_pct = ticker_data.get('todaysChangePerc', 0)

                # Calculate volume surge (simplified)
                # This is just for demonstration - real system would use historical averages
                avg_volume = volume * 0.8  # Simplified baseline
                rvol = volume / avg_volume if avg_volume > 0 else 1.0

                # Simple scoring based on real data
                score = min(100, int(rvol * 10 + abs(change_pct) * 2))

                candidate = {
                    'symbol': symbol,
                    'price': round(price, 2),
                    'volume_surge': round(rvol, 1),
                    'percent_change': round(change_pct, 1),
                    'accumulation_score': score,
                    'tier': 'A-TIER' if score >= 80 else 'B-TIER',
                    'data_source': 'MCP_REAL'
                }

                results.append(candidate)
                print(f"✅ {symbol}: ${price:.2f} | {rvol:.1f}x RVOL | {score} score | {candidate['tier']}")

        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")

    print(f"\n📊 MCP Discovery Results: {len(results)} candidates found")
    return results

# This function is designed to be called from Claude Code where MCP functions exist
if __name__ == "__main__":
    print("⚠️  This script requires MCP functions available in Claude Code environment")
    print("    Run this via Claude Code, not standalone Python")