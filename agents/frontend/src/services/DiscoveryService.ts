import { StockAnalysis, MarketMetrics } from '../types/trading';

export class DiscoveryService {
  private static baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:3001';

  /**
   * Fetch explosive stock recommendations from the discovery system
   */
  static async getExplosiveStocks(): Promise<StockAnalysis[]> {
    try {
      console.log('🚀 Fetching from discovery API...');
      const response = await fetch(`https://alphastack-discovery.onrender.com/signals/top`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      console.log(`📡 Discovery API response: ${response.status} ${response.statusText}`);

      if (!response.ok) {
        throw new Error(`Discovery API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log(`✅ Discovery API returned ${data.length} stocks:`, data.map((s: any) => s.symbol));

      // Validate data structure
      if (!Array.isArray(data)) {
        console.error('❌ Discovery API returned non-array data:', data);
        throw new Error('Invalid data format from discovery API');
      }

      // Transform discovery API data to frontend format
      const transformedData = data.map((stock: any) => ({
        symbol: stock.symbol,
        name: stock.symbol, // Discovery API doesn't provide company names
        price: parseFloat(stock.price),
        change: 0, // Not provided by discovery API
        change_percent: 0, // Not provided by discovery API
        volume: parseInt(stock.volume || 0),
        market_cap: null, // Not provided by discovery API
        pe_ratio: null, // Not provided by discovery API
        short_interest: null, // Not provided by discovery API
        volatility: parseFloat(stock.rvol || 0), // Use RVOL as volatility proxy
        momentum_score: parseFloat(stock.score || 70), // Use discovery score
        volume_score: parseFloat(stock.rvol || 1), // Use RVOL as volume score
        ai_score: Math.round(parseFloat(stock.score || 70)), // Use discovery score as AI score
        signals: [stock.reason || 'Discovery Signal'], // Use reason as signal
        recommendation: 'BUY' as const // Discovery API only returns buy candidates
      }));

      console.log(`🎯 Transformed ${transformedData.length} stocks for frontend`);
      return transformedData;

    } catch (error) {
      console.error('❌ Discovery API failed:', error);

      // Skip Redis fallback and just return empty array with clear logging
      console.log('⚠️ Returning empty array due to discovery failure');
      return [];
    }
  }

  /**
   * Get market overview metrics
   */
  static async getMarketMetrics(): Promise<MarketMetrics> {
    try {
      const response = await fetch(`${this.baseUrl}/api/market/metrics`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`Market metrics API error: ${response.status}`);
      }

      return await response.json();

    } catch (error) {
      console.warn('Failed to fetch market metrics:', error);

      // Return default metrics if API fails
      return {
        totalStocksScanned: 0,
        explosiveOpportunities: 0,
        averageVolatility: 0.3,
        volumeSurges: 0,
        marketSentiment: 'neutral',
        sectorLeaders: []
      };
    }
  }

  /**
   * Fallback method to get stocks from Redis cache
   */
  private static async getExplosiveStocksFromRedis(): Promise<StockAnalysis[]> {
    const response = await fetch(`${this.baseUrl}/api/cache/explosive-stocks`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      throw new Error('Redis cache unavailable');
    }

    const data = await response.json();
    return data.stocks || [];
  }

  /**
   * Get real-time stock quote for a specific symbol
   */
  static async getStockQuote(symbol: string): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/api/stocks/quote/${symbol}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`Quote API error: ${response.status}`);
      }

      return await response.json();

    } catch (error) {
      console.error(`Failed to fetch quote for ${symbol}:`, error);
      throw error;
    }
  }

  /**
   * Update screening criteria for the discovery system
   */
  static async updateScreeningCriteria(criteria: any): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/api/discovery/criteria`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(criteria)
      });

      if (!response.ok) {
        throw new Error(`Failed to update criteria: ${response.status}`);
      }

    } catch (error) {
      console.error('Failed to update screening criteria:', error);
      throw error;
    }
  }

  /**
   * Get historical performance of discovered stocks
   */
  static async getDiscoveryPerformance(days: number = 30): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/api/discovery/performance?days=${days}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`Performance API error: ${response.status}`);
      }

      return await response.json();

    } catch (error) {
      console.error('Failed to fetch discovery performance:', error);
      throw error;
    }
  }

  /**
   * Subscribe to real-time discovery updates via WebSocket
   */
  static createWebSocketConnection(
    onStockUpdate: (stock: StockAnalysis) => void,
    onMarketUpdate: (metrics: MarketMetrics) => void,
    onError: (error: Error) => void
  ): WebSocket | null {
    try {
      const wsUrl = `${this.baseUrl.replace('http', 'ws')}/ws/discovery`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('Discovery WebSocket connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          switch (data.type) {
            case 'stock_update':
              onStockUpdate(data.stock);
              break;
            case 'market_update':
              onMarketUpdate(data.metrics);
              break;
            case 'heartbeat':
              // Keep connection alive
              break;
            default:
              console.log('Unknown WebSocket message:', data);
          }

        } catch (error) {
          console.error('WebSocket message parsing error:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('Discovery WebSocket error:', error);
        onError(new Error('WebSocket connection failed'));
      };

      ws.onclose = () => {
        console.log('Discovery WebSocket disconnected');
        // Auto-reconnect logic could be added here
      };

      return ws;

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      onError(new Error(`WebSocket creation failed: ${(error as any).message}`));
      return null;
    }
  }

  /**
   * Test connection to discovery system
   */
  static async testConnection(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      return response.ok;

    } catch (error) {
      console.error('Discovery system health check failed:', error);
      return false;
    }
  }

  /**
   * Get system status and agent information
   */
  static async getSystemStatus(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/api/system/status`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`System status API error: ${response.status}`);
      }

      return await response.json();

    } catch (error) {
      console.error('Failed to fetch system status:', error);
      throw error;
    }
  }
}