export interface AgentStatus {
  name: string;
  status: 'online' | 'offline' | 'busy';
  lastHeartbeat: string;
  currentTask?: string;
  performanceMetrics?: Record<string, any>;
}

export interface SystemMetrics {
  totalAgents: number;
  activeAgents: number;
  totalTrades: number;
  portfolioValue: number;
  dailyPnl: number;
  systemHealth: 'healthy' | 'degraded' | 'critical';
}

export interface Stock {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap?: number;
  pe?: number;
  shortInterest?: number;
  aiScore?: number;
  signals?: string[];
}

export interface Position {
  symbol: string;
  shares: number;
  avgPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
  marketValue: number;
}

export interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  timestamp: string;
  status: 'pending' | 'filled' | 'cancelled';
  strategy?: string;
}

export interface BacktestResult {
  strategy: string;
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  period: string;
}

export interface Alert {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  message: string;
  timestamp: string;
  agent?: string;
}