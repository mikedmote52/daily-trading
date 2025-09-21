// Real stock data service using yfinance backend
import { useState, useEffect } from 'react';

// Popular tech stocks for explosive discovery
const DISCOVERY_SYMBOLS = [
  'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX',
  'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'NOW', 'SHOP', 'SQ',
  'ROKU', 'PLTR', 'SNOW', 'ZM', 'DOCU', 'TWLO', 'OKTA', 'DDOG'
];

class StockDataService {
  constructor() {
    this.cache = new Map();
    this.cacheTimeout = 30000; // 30 seconds
  }

  async fetchStockData(symbol) {
    const cacheKey = symbol;
    const cached = this.cache.get(cacheKey);

    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }

    try {
      // Use Yahoo Finance API (free, no key required)
      const response = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/${symbol}`);

      if (!response.ok) {
        throw new Error(`Failed to fetch ${symbol}: ${response.status}`);
      }

      const data = await response.json();
      const result = data.chart.result[0];

      if (!result) {
        throw new Error(`No data found for ${symbol}`);
      }

      const meta = result.meta;
      const quote = result.indicators.quote[0];
      const timestamps = result.timestamp;

      // Calculate volume surge and volatility
      const volumes = quote.volume.filter(v => v !== null);
      const closes = quote.close.filter(c => c !== null);

      const avgVolume = volumes.slice(-20).reduce((a, b) => a + b, 0) / 20;
      const currentVolume = volumes[volumes.length - 1] || 0;
      const volumeRatio = currentVolume / avgVolume;

      // Calculate recent volatility
      const recentCloses = closes.slice(-10);
      const returns = recentCloses.slice(1).map((close, i) =>
        Math.log(close / recentCloses[i])
      );
      const volatility = Math.sqrt(
        returns.reduce((sum, ret) => sum + ret * ret, 0) / returns.length
      ) * Math.sqrt(252) * 100; // Annualized %

      const stockData = {
        symbol: symbol,
        price: meta.regularMarketPrice,
        change: meta.regularMarketPrice - meta.previousClose,
        changePercent: ((meta.regularMarketPrice - meta.previousClose) / meta.previousClose) * 100,
        volume: currentVolume,
        avgVolume: avgVolume,
        volumeRatio: volumeRatio,
        volatility: volatility,
        marketCap: meta.marketCap,
        // Generate explosive score based on volume and volatility
        score: this.calculateExplosiveScore(volumeRatio, volatility, meta.regularMarketPrice - meta.previousClose),
        timestamp: Date.now()
      };

      this.cache.set(cacheKey, { data: stockData, timestamp: Date.now() });
      return stockData;

    } catch (error) {
      console.error(`Error fetching ${symbol}:`, error);
      throw error;
    }
  }

  calculateExplosiveScore(volumeRatio, volatility, priceChange) {
    // Score based on volume surge, volatility, and momentum
    let score = 0;

    // Volume component (40% weight)
    if (volumeRatio > 3) score += 40;
    else if (volumeRatio > 2) score += 30;
    else if (volumeRatio > 1.5) score += 20;

    // Volatility component (30% weight)
    if (volatility > 50) score += 30;
    else if (volatility > 30) score += 20;
    else if (volatility > 20) score += 10;

    // Momentum component (30% weight)
    if (priceChange > 0) {
      const momentum = Math.min(Math.abs(priceChange) * 10, 30);
      score += momentum;
    }

    return Math.min(score, 100);
  }

  async getExplosiveStocks() {
    try {
      console.log('🔍 Scanning stocks for explosive opportunities...');

      const results = await Promise.allSettled(
        DISCOVERY_SYMBOLS.map(symbol => this.fetchStockData(symbol))
      );

      const stocks = results
        .filter(result => result.status === 'fulfilled')
        .map(result => result.value)
        .filter(stock => stock.score >= 60) // Only high-scoring opportunities
        .sort((a, b) => b.score - a.score) // Sort by score
        .slice(0, 8) // Top 8 opportunities
        .map(stock => ({
          symbol: stock.symbol,
          price: stock.price,
          score: stock.score,
          volume: stock.volume,
          rvol: stock.volumeRatio,
          reason: this.generateReason(stock)
        }));

      console.log(`✅ Found ${stocks.length} explosive opportunities`);
      return stocks;

    } catch (error) {
      console.error('❌ Error in explosive stock discovery:', error);
      throw error;
    }
  }

  generateReason(stock) {
    const reasons = [];

    if (stock.volumeRatio > 3) {
      reasons.push(`${stock.volumeRatio.toFixed(1)}x volume surge indicating major institutional activity`);
    }

    if (stock.volatility > 40) {
      reasons.push(`high volatility (${stock.volatility.toFixed(1)}%) suggesting explosive potential`);
    }

    if (stock.changePercent > 2) {
      reasons.push(`strong ${stock.changePercent.toFixed(1)}% upward momentum`);
    }

    return `${stock.symbol} shows ${reasons.join(' with ')}.`;
  }
}

export const stockService = new StockDataService();

// React hook for using stock data
export function useExplosiveStocks() {
  const [stocks, setStocks] = useState({ data: [], loading: true, error: null });

  useEffect(() => {
    const fetchStocks = async () => {
      try {
        setStocks(prev => ({ ...prev, loading: true, error: null }));

        // Try MCP-enhanced backend first
        console.log('🔍 Fetching MCP-enhanced stock data...');
        const response = await fetch('http://localhost:8081/stocks/explosive');

        if (!response.ok) {
          throw new Error(`MCP backend error: ${response.status}`);
        }

        const result = await response.json();

        if (result.error) {
          throw new Error(result.error);
        }

        const stocks = result.stocks || [];
        console.log(`✅ Received ${stocks.length} MCP-filtered stocks`);

        setStocks({ data: stocks, loading: false, error: null });

      } catch (error) {
        console.error('❌ MCP backend failed, falling back to original service:', error);

        // Fallback to original stock service
        try {
          const data = await stockService.getExplosiveStocks();
          setStocks({ data, loading: false, error: null });
        } catch (fallbackError) {
          setStocks({ data: [], loading: false, error: fallbackError.message });
        }
      }
    };

    fetchStocks();

    // Refresh every 60 seconds
    const interval = setInterval(fetchStocks, 60000);
    return () => clearInterval(interval);
  }, []);

  return stocks;
}