// Core stock analysis interface
export interface StockAnalysis {
  symbol: string;
  name: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  rvol: number;  // CRITICAL: Relative Volume (most important metric for stealth detection)
  market_cap: number | null;
  pe_ratio: number | null;
  short_interest: number | null;
  volatility: number;
  momentum_score: number;
  volume_score: number;
  ai_score: number;
  signals: string[];
  recommendation: 'BUY' | 'SELL' | 'HOLD' | 'AVOID';

  // Local Web Enrichment Fields
  web_catalyst_summary?: string | null;
  web_catalyst_score?: number;
  web_sentiment_score?: number;
  web_sentiment_description?: string | null;
  institutional_activity?: string | null;
  institutional_score?: number;
  explosion_probability?: number;
}

// Market overview metrics
export interface MarketMetrics {
  totalStocksScanned: number;
  explosiveOpportunities: number;
  averageVolatility: number;
  volumeSurges: number;
  marketSentiment: 'bullish' | 'bearish' | 'neutral';
  sectorLeaders: SectorData[];
}

// Sector performance data
export interface SectorData {
  sector: string;
  performance: number;
  stockCount: number;
  topPerformer: string;
}

// Discovery criteria for screening
export interface DiscoveryScreeningCriteria {
  max_price: number;
  min_price: number;
  min_volume: number;
  volume_surge_threshold: number;
  min_market_cap: number;
  max_market_cap: number;
  min_volatility: number;
  max_volatility: number;
  min_short_interest: number;
  high_short_interest: number;
  momentum_period: number;
  min_momentum_score: number;
  rsi_oversold: number;
  rsi_overbought: number;
  max_pe_ratio: number;
  min_revenue_growth: number;
}

// Trading position interface
export interface TradingPosition {
  symbol: string;
  quantity: number;
  averagePrice: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPL: number;
  unrealizedPLPercent: number;
  dayChange: number;
  dayChangePercent: number;
  entryDate: Date;
  stopLoss?: number;
  takeProfit?: number;
}

// Portfolio overview
export interface Portfolio {
  totalValue: number;
  totalCash: number;
  buyingPower: number;
  dayPL: number;
  dayPLPercent: number;
  totalPL: number;
  totalPLPercent: number;
  positions: TradingPosition[];
}

// Trading order interface
export interface TradingOrder {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  orderType: 'market' | 'limit' | 'stop' | 'stop_limit';
  status: 'new' | 'filled' | 'partially_filled' | 'cancelled' | 'rejected';
  filledQuantity: number;
  averageFillPrice?: number;
  limitPrice?: number;
  stopPrice?: number;
  timeInForce: 'day' | 'gtc' | 'ioc' | 'fok';
  submittedAt: Date;
  filledAt?: Date;
}

// Risk management settings
export interface RiskManagementSettings {
  maxPositionSize: number;
  maxPortfolioRisk: number;
  stopLossPercentage: number;
  takeProfitPercentage: number;
  maxDailyLoss: number;
  maxPositions: number;
  allowDayTrading: boolean;
  positionSizingMethod: 'fixed' | 'kelly' | 'volatility_adjusted';
}

// Real-time stock quote
export interface StockQuote {
  symbol: string;
  price: number;
  bid: number;
  ask: number;
  bidSize: number;
  askSize: number;
  lastTrade: {
    price: number;
    size: number;
    timestamp: Date;
  };
  dailyChange: number;
  dailyChangePercent: number;
  volume: number;
  averageVolume: number;
  high: number;
  low: number;
  open: number;
  previousClose: number;
}

// Technical analysis data
export interface TechnicalAnalysis {
  symbol: string;
  rsi: number;
  macd: {
    macd: number;
    signal: number;
    histogram: number;
  };
  movingAverages: {
    sma20: number;
    sma50: number;
    sma200: number;
    ema12: number;
    ema26: number;
  };
  support: number[];
  resistance: number[];
  patterns: string[];
  trend: 'bullish' | 'bearish' | 'sideways';
}

// Discovery performance tracking
export interface DiscoveryPerformance {
  totalRecommendations: number;
  successfulTrades: number;
  successRate: number;
  averageReturn: number;
  bestTrade: {
    symbol: string;
    return: number;
    duration: number;
  };
  worstTrade: {
    symbol: string;
    return: number;
    duration: number;
  };
  sectorBreakdown: {
    [sector: string]: {
      count: number;
      successRate: number;
      averageReturn: number;
    };
  };
}

// System agent status
export interface AgentStatus {
  name: string;
  status: 'active' | 'inactive' | 'error';
  lastHeartbeat: Date;
  currentTask: string;
  metrics: {
    [key: string]: number | string;
  };
  errorMessage?: string;
}

// Trading alert/notification
export interface TradingAlert {
  id: string;
  type: 'opportunity' | 'risk' | 'execution' | 'system';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  symbol?: string;
  actionRequired: boolean;
  timestamp: Date;
  acknowledged: boolean;
}

// WebSocket message types
export interface WebSocketMessage {
  type: 'stock_update' | 'market_update' | 'order_update' | 'alert' | 'heartbeat';
  data: any;
  timestamp: Date;
}

// Backtesting results
export interface BacktestResults {
  startDate: Date;
  endDate: Date;
  initialCapital: number;
  finalValue: number;
  totalReturn: number;
  totalReturnPercent: number;
  annualizedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  maxDrawdownPercent: number;
  winRate: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  averageWin: number;
  averageLoss: number;
  profitFactor: number;
  trades: BacktestTrade[];
}

// Individual backtest trade
export interface BacktestTrade {
  symbol: string;
  entryDate: Date;
  exitDate: Date;
  entryPrice: number;
  exitPrice: number;
  quantity: number;
  profit: number;
  profitPercent: number;
  duration: number; // days
  reason: string;
}

// API response wrapper
export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  timestamp: Date;
}

// Error response
export interface ApiError {
  success: false;
  error: string;
  code: number;
  timestamp: Date;
}