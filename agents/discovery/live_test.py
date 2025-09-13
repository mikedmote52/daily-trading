#!/usr/bin/env python3
"""
Live test of enhanced explosive growth detection
"""

import asyncio
from main import StockDiscoveryAgent

async def get_top_discoveries():
    agent = StockDiscoveryAgent()
    print('🔍 Starting live explosive growth scan...')
    
    # Get a focused sample of discovered stocks
    discovered = await agent._screen_stocks()
    print(f'📊 Found {len(discovered)} stocks passing initial screening')
    
    # Analyze top candidates
    analyzed = []
    count = 0
    for stock_data in discovered[:15]:  # Check first 15 stocks
        try:
            analysis = await agent._analyze_stock(stock_data)
            if analysis and analysis.ai_score >= 60:  # High-scoring stocks
                analyzed.append(analysis)
                count += 1
                print(f'✅ {analysis.symbol}: Score {analysis.ai_score} - {analysis.recommendation}')
                if count >= 8:  # Show top 8
                    break
        except Exception as e:
            continue
    
    # Sort by score and show results
    analyzed.sort(key=lambda x: x.ai_score, reverse=True)
    
    print('\n🚀 TOP EXPLOSIVE GROWTH CANDIDATES:')
    print('=' * 60)
    for i, stock in enumerate(analyzed[:5], 1):
        print(f'{i}. {stock.symbol} - ${stock.price:.2f}')
        print(f'   📈 Explosive Score: {stock.ai_score}/100')
        print(f'   💰 Market Cap: ${stock.market_cap/1e9:.1f}B')
        print(f'   📊 Momentum: {stock.momentum_score:.1f}%')
        print(f'   🔊 Volume Surge: {stock.volume_score:.1f}x')
        print(f'   ⚡ Short Interest: {stock.short_interest:.1f}%')
        print(f'   🎯 Recommendation: {stock.recommendation}')
        print(f'   📋 Signals: {", ".join(stock.signals[:3])}')
        print()

if __name__ == "__main__":
    asyncio.run(get_top_discoveries())