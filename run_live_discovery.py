#!/usr/bin/env python3
"""
LIVE UNIVERSAL DISCOVERY - Full Market Scan
Shows actual filtering process on entire stock universe
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents', 'discovery'))

from universal_discovery import UniversalDiscoverySystem
import json
from datetime import datetime

def main():
    print("="*80)
    print("🚀 LIVE UNIVERSAL DISCOVERY SYSTEM - FULL MARKET SCAN")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Initialize the discovery system
    discovery = UniversalDiscoverySystem()

    # Run the full discovery pipeline
    print("\n📡 Starting full universe scan...")
    print("   This will process ~5,200 stocks through all gates")
    print("   Expected time: 60-120 seconds\n")

    result = discovery.run_universal_discovery()

    # Display detailed results
    if result:
        print("\n" + "="*80)
        print("📊 DISCOVERY PIPELINE RESULTS")
        print("="*80)

        # Universe coverage stats
        coverage = result.get('universe_coverage', {})
        print(f"\n🌍 UNIVERSE COVERAGE:")
        print(f"   Total Universe Scanned: {coverage.get('total_universe', 0):,} stocks")
        print(f"   After Gate A (Initial): {coverage.get('gate_a_output', 0):,} stocks")
        print(f"   After Gate B (Fundamental): {coverage.get('gate_b_output', 0):,} stocks")
        print(f"   After Gate C (Final): {coverage.get('final_candidates', 0):,} stocks")

        # Results summary
        summary = result.get('results_summary', {})
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Trade Ready: {summary.get('trade_ready_count', 0)} stocks")
        print(f"   Watchlist: {summary.get('watchlist_count', 0)} stocks")
        print(f"   Processing Time: {result.get('processing_time_seconds', 0):.1f} seconds")

        # Top candidates
        results = result.get('results', [])
        if results:
            print(f"\n🔥 TOP EXPLOSIVE CANDIDATES:")
            print("-"*80)

            # Show top 10 candidates
            for i, stock in enumerate(results[:10], 1):
                print(f"\n{i}. {stock['symbol']} - Score: {stock['accumulation_score']}/100 - {stock['status']}")
                print(f"   Price: ${stock['price']:.2f}")
                print(f"   Volume Surge: {stock['volume_surge']:.1f}x")
                print(f"   Market Cap: ${stock['market_cap_billions']:.1f}B")
                print(f"   Short Interest: {stock['short_interest']:.1f}%")
                print(f"   IV Percentile: {stock['iv_percentile']:.0f}%")

                # Show bucket scores
                buckets = stock.get('bucket_scores', {})
                if buckets:
                    print(f"   Scoring Breakdown:")
                    print(f"      Volume Pattern: {buckets.get('volume_pattern', 0)}/100")
                    print(f"      Float/Short: {buckets.get('float_short', 0)}/100")
                    print(f"      Options Activity: {buckets.get('options_activity', 0)}/100")
                    print(f"      Technical Setup: {buckets.get('technical_setup', 0)}/100")

                # Show warnings
                warnings = stock.get('warnings', [])
                if warnings:
                    print(f"   ⚠️ Warnings: {', '.join(warnings)}")
        else:
            print("\n❌ No candidates passed all filters")

        # Save full results to file
        output_file = f"discovery_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Full results saved to: {output_file}")
    else:
        print("\n❌ Discovery system returned no results")

if __name__ == "__main__":
    main()