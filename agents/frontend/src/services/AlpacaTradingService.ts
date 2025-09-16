import { StockAnalysis } from '../types/trading';

export interface AlpacaOrder {
  id: string;
  symbol: string;
  qty: number;
  side: 'buy' | 'sell';
  order_type: string;
  status: string;
  filled_price?: number;
  stop_loss?: number;
  take_profit?: number;
}

export interface Position {
  symbol: string;
  qty: number;
  cost_basis: number;
  market_value: number;
  unrealized_pl: number;
  unrealized_plpc: number;
}

export interface AccountInfo {
  cash: number;
  portfolio_value: number;
  buying_power: number;
  day_trade_count: number;
  equity: number;
}

export class AlpacaTradingService {
  private apiKey: string;
  private secretKey: string;
  private baseUrl: string;
  private isLiveTrading: boolean;

  constructor(
    apiKey: string,
    secretKey: string,
    isLiveTrading: boolean = false
  ) {
    this.apiKey = apiKey;
    this.secretKey = secretKey;
    this.isLiveTrading = isLiveTrading;
    this.baseUrl = isLiveTrading
      ? 'https://api.alpaca.markets'
      : 'https://paper-api.alpaca.markets';
  }

  private getHeaders(): HeadersInit {
    return {
      'APCA-API-KEY-ID': this.apiKey,
      'APCA-API-SECRET-KEY': this.secretKey,
      'Content-Type': 'application/json',
    };
  }

  /**
   * Execute explosive stock trade with intelligent position sizing and risk management
   */
  async executeExplosiveTrade(
    stockAnalysis: StockAnalysis,
    investmentAmount: number
  ): Promise<AlpacaOrder> {
    try {
      // Get account info for position sizing validation
      const accountInfo = await this.getAccountInfo();

      // Validate buying power
      if (investmentAmount > accountInfo.buying_power) {
        throw new Error(`Insufficient buying power. Available: $${accountInfo.buying_power.toFixed(2)}`);
      }

      // Calculate position size
      const shares = Math.floor(investmentAmount / stockAnalysis.price);

      if (shares === 0) {
        throw new Error(`Investment amount too small. Minimum: $${stockAnalysis.price.toFixed(2)}`);
      }

      // Calculate risk management levels
      const stopLossPrice = this.calculateStopLoss(stockAnalysis);
      const takeProfitPrice = this.calculateTakeProfit(stockAnalysis);

      // Place main order
      const mainOrder = await this.placeOrder({
        symbol: stockAnalysis.symbol,
        qty: shares,
        side: 'buy',
        type: 'market',
        time_in_force: 'day'
      });

      // Place bracket order with stop-loss and take-profit
      if (mainOrder.status === 'filled' || mainOrder.status === 'partially_filled') {
        // Place stop-loss order
        await this.placeOrder({
          symbol: stockAnalysis.symbol,
          qty: shares,
          side: 'sell',
          type: 'stop',
          time_in_force: 'gtc',
          stop_price: stopLossPrice
        });

        // Place take-profit order
        await this.placeOrder({
          symbol: stockAnalysis.symbol,
          qty: shares,
          side: 'sell',
          type: 'limit',
          time_in_force: 'gtc',
          limit_price: takeProfitPrice
        });
      }

      return {
        ...mainOrder,
        stop_loss: stopLossPrice,
        take_profit: takeProfitPrice
      };

    } catch (error) {
      console.error('Alpaca trading error:', error);
      throw new Error(`Trading failed: ${error.message}`);
    }
  }

  /**
   * Place order with Alpaca API
   */
  private async placeOrder(orderData: any): Promise<AlpacaOrder> {
    const response = await fetch(`${this.baseUrl}/v2/orders`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(orderData)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || 'Order placement failed');
    }

    return await response.json();
  }

  /**
   * Calculate stop-loss price based on volatility and AI confidence
   */
  private calculateStopLoss(stockAnalysis: StockAnalysis): number {
    // Base stop-loss at 2x daily volatility or 5%, whichever is larger
    const volatilityStop = stockAnalysis.price * (1 - Math.max(stockAnalysis.volatility * 2 / 252, 0.05));

    // Adjust based on AI confidence - higher confidence allows tighter stops
    const confidenceMultiplier = stockAnalysis.ai_score >= 80 ? 0.8 : 1.0;

    return Math.round(volatilityStop * confidenceMultiplier * 100) / 100;
  }

  /**
   * Calculate take-profit price based on explosive potential
   */
  private calculateTakeProfit(stockAnalysis: StockAnalysis): number {
    // Base target: AI score percentage of the price (capped at 50%)
    let targetMultiplier = Math.min(stockAnalysis.ai_score / 100 * 0.5, 0.5);

    // Bonus for specific explosive patterns
    if (stockAnalysis.signals.includes('Volume Surge') && stockAnalysis.volume_score > 3) {
      targetMultiplier += 0.1;
    }

    if (stockAnalysis.short_interest && stockAnalysis.short_interest > 20) {
      targetMultiplier += 0.15; // Short squeeze bonus
    }

    if (stockAnalysis.momentum_score > 15) {
      targetMultiplier += 0.1; // Strong momentum bonus
    }

    // Conservative cap at 75% gain
    targetMultiplier = Math.min(targetMultiplier, 0.75);

    return Math.round(stockAnalysis.price * (1 + targetMultiplier) * 100) / 100;
  }

  /**
   * Get current account information
   */
  async getAccountInfo(): Promise<AccountInfo> {
    const response = await fetch(`${this.baseUrl}/v2/account`, {
      method: 'GET',
      headers: this.getHeaders()
    });

    if (!response.ok) {
      throw new Error('Failed to fetch account information');
    }

    const data = await response.json();
    return {
      cash: parseFloat(data.cash),
      portfolio_value: parseFloat(data.portfolio_value),
      buying_power: parseFloat(data.buying_power),
      day_trade_count: parseInt(data.day_trade_count),
      equity: parseFloat(data.equity)
    };
  }

  /**
   * Get current positions
   */
  async getPositions(): Promise<Position[]> {
    const response = await fetch(`${this.baseUrl}/v2/positions`, {
      method: 'GET',
      headers: this.getHeaders()
    });

    if (!response.ok) {
      throw new Error('Failed to fetch positions');
    }

    const data = await response.json();
    return data.map((position: any) => ({
      symbol: position.symbol,
      qty: parseInt(position.qty),
      cost_basis: parseFloat(position.cost_basis),
      market_value: parseFloat(position.market_value),
      unrealized_pl: parseFloat(position.unrealized_pl),
      unrealized_plpc: parseFloat(position.unrealized_plpc)
    }));
  }

  /**
   * Get recent orders
   */
  async getOrders(limit: number = 50): Promise<AlpacaOrder[]> {
    const response = await fetch(`${this.baseUrl}/v2/orders?limit=${limit}&status=all`, {
      method: 'GET',
      headers: this.getHeaders()
    });

    if (!response.ok) {
      throw new Error('Failed to fetch orders');
    }

    return await response.json();
  }

  /**
   * Cancel all open orders for a symbol
   */
  async cancelOrdersForSymbol(symbol: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/v2/orders`, {
      method: 'DELETE',
      headers: this.getHeaders(),
      body: JSON.stringify({ symbol })
    });

    if (!response.ok) {
      throw new Error(`Failed to cancel orders for ${symbol}`);
    }
  }

  /**
   * Close position (market sell)
   */
  async closePosition(symbol: string): Promise<AlpacaOrder> {
    const position = await this.getPosition(symbol);

    if (!position) {
      throw new Error(`No position found for ${symbol}`);
    }

    return await this.placeOrder({
      symbol: symbol,
      qty: Math.abs(position.qty),
      side: position.qty > 0 ? 'sell' : 'buy',
      type: 'market',
      time_in_force: 'day'
    });
  }

  private async getPosition(symbol: string): Promise<Position | null> {
    try {
      const response = await fetch(`${this.baseUrl}/v2/positions/${symbol}`, {
        method: 'GET',
        headers: this.getHeaders()
      });

      if (!response.ok) {
        return null;
      }

      const data = await response.json();
      return {
        symbol: data.symbol,
        qty: parseInt(data.qty),
        cost_basis: parseFloat(data.cost_basis),
        market_value: parseFloat(data.market_value),
        unrealized_pl: parseFloat(data.unrealized_pl),
        unrealized_plpc: parseFloat(data.unrealized_plpc)
      };
    } catch {
      return null;
    }
  }
}