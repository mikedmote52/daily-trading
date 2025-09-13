#!/usr/bin/env python3
"""
Trading System Type Definitions

Common data types used across all agents.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from enum import Enum

# Enums
class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class AgentStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"

class AlertType(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"

class SystemHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"

# Data Classes
@dataclass
class Stock:
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    short_interest: Optional[float] = None
    volatility: Optional[float] = None
    ai_score: Optional[int] = None
    signals: List[str] = None
    sector: Optional[str] = None
    
    def __post_init__(self):
        if self.signals is None:
            self.signals = []

@dataclass
class Position:
    symbol: str
    shares: int
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    weight: float
    entry_date: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class Trade:
    id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime
    status: OrderStatus
    strategy: Optional[str] = None
    commission: Optional[float] = None
    notes: Optional[str] = None

@dataclass
class TradingSignal:
    symbol: str
    signal_type: str
    side: OrderSide
    strength: float  # 0-1 scale
    confidence: float  # 0-1 scale
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    description: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class BacktestParameters:
    start_date: str
    end_date: str
    initial_capital: float
    max_positions: int
    stop_loss: float
    take_profit: float
    rebalance_frequency: str
    commission: float = 0.001

@dataclass
class BacktestResult:
    strategy: str
    parameters: BacktestParameters
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    calmar_ratio: float
    sortino_ratio: float
    period: str
    equity_curve: List[Dict[str, Any]]
    trade_log: List[Dict[str, Any]]
    monthly_returns: List[float]

@dataclass
class RiskMetrics:
    var_95: float  # Value at Risk (95% confidence)
    cvar_95: float  # Conditional Value at Risk
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    beta: float
    correlation_spy: float
    portfolio_concentration: float
    sector_concentration: Dict[str, float]

@dataclass
class PortfolioAllocation:
    symbol: str
    target_weight: float
    current_weight: float
    shares_to_trade: int
    trade_value: float
    action: str  # 'BUY', 'SELL', 'HOLD'
    rationale: str = ""

@dataclass
class Alert:
    id: str
    type: AlertType
    message: str
    timestamp: datetime
    agent: Optional[str] = None
    severity: Optional[str] = None
    acknowledged: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class AgentMetrics:
    agent_name: str
    status: AgentStatus
    last_heartbeat: datetime
    current_task: Optional[str] = None
    performance_metrics: Dict[str, Any] = None
    error_count: int = 0
    uptime: float = 0.0  # Hours
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}

@dataclass
class SystemMetrics:
    total_agents: int
    active_agents: int
    total_trades: int
    portfolio_value: float
    daily_pnl: float
    system_health: SystemHealth
    error_count: int = 0
    uptime: float = 0.0  # Hours
    last_update: datetime = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.now()

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    prev_close: Optional[float] = None

@dataclass
class ScreeningCriteria:
    max_price: float = 100.0
    min_volume: int = 1000000
    min_market_cap: float = 1000000000
    max_pe_ratio: float = 30.0
    min_short_interest: float = 5.0
    min_volatility: float = 0.02
    momentum_period: int = 20
    volume_surge_threshold: float = 2.0
    sectors_include: List[str] = None
    sectors_exclude: List[str] = None
    
    def __post_init__(self):
        if self.sectors_include is None:
            self.sectors_include = []
        if self.sectors_exclude is None:
            self.sectors_exclude = []

@dataclass
class StrategyConfig:
    name: str
    parameters: Dict[str, Any]
    enabled: bool = True
    risk_multiplier: float = 1.0
    max_allocation: float = 0.30
    description: str = ""

@dataclass
class APICredentials:
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    polygon_api_key: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None
    claude_api_key: str = ""

# Type Aliases
TradeAction = Literal["BUY", "SELL", "HOLD"]
RebalanceFrequency = Literal["daily", "weekly", "monthly", "quarterly"]
StrategyType = Literal["momentum", "mean_reversion", "breakout", "ai_signals", "pairs_trading"]

# Helper Functions
def create_alert(alert_type: AlertType, message: str, agent: str = None) -> Alert:
    """Create a new alert"""
    import uuid
    return Alert(
        id=str(uuid.uuid4()),
        type=alert_type,
        message=message,
        timestamp=datetime.now(),
        agent=agent
    )

def calculate_position_value(shares: int, price: float) -> float:
    """Calculate position market value"""
    return shares * price

def calculate_pnl(shares: int, current_price: float, avg_price: float) -> Tuple[float, float]:
    """Calculate unrealized P&L"""
    unrealized_pnl = shares * (current_price - avg_price)
    unrealized_pnl_percent = (current_price - avg_price) / avg_price * 100 if avg_price > 0 else 0.0
    return unrealized_pnl, unrealized_pnl_percent

def validate_trade_parameters(symbol: str, shares: int, price: float) -> bool:
    """Validate trade parameters"""
    if not symbol or not symbol.isalpha():
        return False
    if shares <= 0:
        return False
    if price <= 0:
        return False
    return True

def format_currency(amount: float) -> str:
    """Format currency amount"""
    return f"${amount:,.2f}"

def format_percentage(value: float) -> str:
    """Format percentage value"""
    return f"{value:+.2f}%"