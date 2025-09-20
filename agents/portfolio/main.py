#!/usr/bin/env python3
"""
Portfolio Management Agent

Constructs, monitors, and optimizes the AI-managed stock portfolio.
Handles position sizing, risk management, and execution via Alpaca API.
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import redis
import pandas as pd
import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
import yfinance as yf
from scipy.optimize import minimize
import httpx
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PortfolioAgent')

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

@dataclass
class RiskMetrics:
    var_95: float  # Value at Risk (95% confidence)
    cvar_95: float  # Conditional Value at Risk
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    beta: float
    correlation_spy: float

@dataclass
class PortfolioAllocation:
    symbol: str
    target_weight: float
    current_weight: float
    shares_to_trade: int
    trade_value: float
    action: str  # 'BUY', 'SELL', 'HOLD'

class RiskManager:
    """Risk management and position sizing"""
    
    def __init__(self, max_position_size: float = 0.10, max_sector_exposure: float = 0.30):
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.var_lookback = 252  # 1 year for VaR calculation
    
    def calculate_position_size(self, portfolio_value: float, signal_strength: float,
                              volatility: float, correlation: float) -> float:
        """Calculate optimal position size using Kelly Criterion modified for risk"""
        
        # Base Kelly fraction (simplified)
        win_prob = 0.5 + (signal_strength / 200)  # Convert signal to probability
        avg_win = 0.05  # 5% average win
        avg_loss = 0.03  # 3% average loss
        
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        
        # Risk adjustment
        volatility_penalty = min(volatility / 0.20, 2.0)  # Penalize high volatility
        correlation_penalty = min(abs(correlation), 1.0)  # Penalize high correlation
        
        adjusted_fraction = kelly_fraction / (volatility_penalty * (1 + correlation_penalty))
        
        # Cap at maximum position size
        final_fraction = min(adjusted_fraction, self.max_position_size)
        
        return max(final_fraction, 0.01)  # Minimum 1% allocation

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk"""
        if len(returns) < 30:
            return 0.0, 0.0
        
        # Historical method
        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var].mean()
        
        return abs(var), abs(cvar)

class PortfolioOptimizer:
    """Portfolio optimization using Modern Portfolio Theory"""
    
    def __init__(self):
        self.lookback_period = 252  # 1 year
        self.min_weight = 0.01
        self.max_weight = 0.15
    
    def optimize_weights(self, expected_returns: pd.Series, 
                        covariance_matrix: pd.DataFrame) -> pd.Series:
        """Optimize portfolio weights using mean-variance optimization"""
        
        n_assets = len(expected_returns)
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(covariance_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = tuple([(self.min_weight, self.max_weight) for _ in range(n_assets)])
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return pd.Series(result.x, index=expected_returns.index)
        else:
            # Fallback to equal weights
            return pd.Series([1/n_assets] * n_assets, index=expected_returns.index)

class PortfolioManagementAgent:
    def __init__(self):
        self.claude = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
        self.redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))

        # Alpaca configuration
        self.alpaca_base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        self.alpaca_key = os.getenv('ALPACA_KEY')
        self.alpaca_secret = os.getenv('ALPACA_SECRET')
        self.orders_api_base = os.getenv('ORDERS_API_URL', 'https://alphastack-orders.onrender.com')

        if not self.alpaca_key or not self.alpaca_secret:
            raise ValueError("ALPACA_KEY and ALPACA_SECRET environment variables are required")

        # Portfolio state (will be loaded from real Alpaca account)
        self.cash = 0.0
        self.positions: Dict[str, Position] = {}
        self.portfolio_value = 0.0
        self.daily_pnl = 0.0

        # Risk management
        self.risk_manager = RiskManager()
        self.optimizer = PortfolioOptimizer()

        # Configuration
        self.auto_execution = True  # Enable real trading
        self.max_positions = 10
        self.rebalance_threshold = 0.05  # 5% deviation triggers rebalance

        self.running = False

    def _get_alpaca_headers(self):
        """Get Alpaca authentication headers"""
        return {
            "APCA-API-KEY-ID": self.alpaca_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret,
            "Content-Type": "application/json"
        }

    async def _initialize_real_portfolio(self):
        """Initialize portfolio from real Alpaca account data"""
        try:
            async with httpx.AsyncClient() as client:
                # Get real account data
                account_response = await client.get(
                    f"{self.alpaca_base}/v2/account",
                    headers=self._get_alpaca_headers()
                )

                if account_response.status_code != 200:
                    raise Exception(f"Failed to fetch account: {account_response.status_code} - {account_response.text}")

                account = account_response.json()
                self.cash = float(account['cash'])
                self.portfolio_value = float(account['portfolio_value'])

                logger.info(f"Loaded real account: ${self.portfolio_value:.2f} portfolio value, ${self.cash:.2f} cash")

                # Get real positions
                positions_response = await client.get(
                    f"{self.alpaca_base}/v2/positions",
                    headers=self._get_alpaca_headers()
                )

                if positions_response.status_code == 200:
                    positions = positions_response.json()

                    for pos in positions:
                        if float(pos['qty']) != 0:  # Only include non-zero positions
                            shares = int(float(pos['qty']))
                            avg_price = float(pos['avg_entry_price'])
                            market_value = float(pos['market_value'])
                            current_price = market_value / abs(shares) if shares != 0 else avg_price

                            position = Position(
                                symbol=pos['symbol'],
                                shares=shares,
                                avg_price=avg_price,
                                current_price=current_price,
                                market_value=market_value,
                                unrealized_pnl=float(pos['unrealized_pl']),
                                unrealized_pnl_percent=float(pos['unrealized_plpc']) * 100,
                                weight=market_value / self.portfolio_value if self.portfolio_value > 0 else 0,
                                entry_date=datetime.now()  # Alpaca doesn't provide entry date in positions
                            )

                            self.positions[pos['symbol']] = position
                            logger.info(f"Loaded position: {pos['symbol']} - {shares} shares @ ${avg_price:.2f}")

                    logger.info(f"Loaded {len(self.positions)} real positions")
                else:
                    logger.warning(f"Failed to fetch positions: {positions_response.status_code}")

        except Exception as e:
            logger.error(f"Failed to initialize real portfolio: {e}")
            raise  # Fail hard - no fallback to fake data

    async def start(self):
        """Start the portfolio management agent"""
        logger.info("Starting Portfolio Management Agent...")

        # Initialize with real portfolio data
        await self._initialize_real_portfolio()

        self.running = True

        # Start background tasks
        tasks = [
            asyncio.create_task(self._monitoring_loop()),
            asyncio.create_task(self._rebalancing_loop()),
            asyncio.create_task(self._risk_monitoring_loop()),
            asyncio.create_task(self._heartbeat_loop())
        ]

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down Portfolio Management Agent...")
            self.running = False

    async def _monitoring_loop(self):
        """Monitor portfolio performance and positions"""
        while self.running:
            try:
                # Update position values
                await self._update_positions()
                
                # Calculate portfolio metrics
                self._update_portfolio_metrics()
                
                # Check for risk violations
                await self._check_risk_limits()
                
                # Update Redis with current state
                await self._update_portfolio_state()
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)

    async def _update_positions(self):
        """Update current position values"""
        for symbol, position in self.positions.items():
            try:
                # Get current price
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    position.current_price = current_price
                    position.market_value = position.shares * current_price
                    position.unrealized_pnl = position.shares * (current_price - position.avg_price)
                    position.unrealized_pnl_percent = (current_price - position.avg_price) / position.avg_price * 100
                    
            except Exception as e:
                logger.warning(f"Failed to update position for {symbol}: {e}")

    def _update_portfolio_metrics(self):
        """Update portfolio-level metrics"""
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        self.portfolio_value = self.cash + total_market_value
        
        # Update position weights
        if self.portfolio_value > 0:
            for position in self.positions.values():
                position.weight = position.market_value / self.portfolio_value
        
        # Calculate daily P&L
        self.daily_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

    async def _rebalancing_loop(self):
        """Periodic portfolio rebalancing"""
        while self.running:
            try:
                # Rebalance every hour during market hours
                current_time = datetime.now()
                if 9 <= current_time.hour <= 16 and current_time.weekday() < 5:  # Market hours
                    
                    # Get trade ideas from discovery agent
                    trade_ideas = await self._get_trade_ideas()
                    
                    if trade_ideas:
                        # Generate portfolio allocation
                        allocation = await self._generate_allocation(trade_ideas)
                        
                        # Check if rebalancing is needed
                        if self._needs_rebalancing(allocation):
                            await self._execute_rebalancing(allocation)
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in rebalancing loop: {e}")
                await asyncio.sleep(3600)

    async def _get_trade_ideas(self) -> List[Dict[str, Any]]:
        """Get trade ideas from discovery agent"""
        try:
            discoveries_data = self.redis_client.get('top_stock_picks')
            if discoveries_data:
                return json.loads(discoveries_data)
            return []
        except Exception as e:
            logger.error(f"Error getting trade ideas: {e}")
            return []

    async def _generate_allocation(self, trade_ideas: List[Dict[str, Any]]) -> List[PortfolioAllocation]:
        """Generate optimal portfolio allocation using Claude"""
        try:
            # Prepare current portfolio state
            current_portfolio = {
                'cash': self.cash,
                'portfolio_value': self.portfolio_value,
                'positions': [asdict(pos) for pos in self.positions.values()],
                'daily_pnl': self.daily_pnl
            }
            
            prompt = f"""
            Generate an optimal portfolio allocation based on:
            
            Current Portfolio:
            {json.dumps(current_portfolio, indent=2, default=str)}
            
            Trade Ideas (top AI-scored stocks):
            {json.dumps(trade_ideas[:10], indent=2, default=str)}
            
            Constraints:
            - Maximum 10 positions
            - Maximum 15% in any single position
            - Minimum 1% position size
            - Total allocation should not exceed available capital
            - Consider risk-adjusted returns
            
            Return a JSON list of allocations:
            [
                {{
                    "symbol": "AAPL",
                    "target_weight": 0.12,
                    "rationale": "Strong AI score and momentum"
                }}
            ]
            
            Focus on diversification and risk management.
            """
            
            response = self.claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            claude_allocation = json.loads(response.content[0].text)
            
            # Convert to PortfolioAllocation objects
            allocations = []
            for alloc in claude_allocation:
                symbol = alloc['symbol']
                target_weight = alloc['target_weight']
                current_weight = self.positions[symbol].weight if symbol in self.positions else 0
                
                # Calculate shares to trade
                target_value = target_weight * self.portfolio_value
                current_value = current_weight * self.portfolio_value
                trade_value = target_value - current_value
                
                # Get current price for share calculation
                try:
                    ticker = yf.Ticker(symbol)
                    current_price = ticker.history(period="1d")['Close'].iloc[-1]
                    shares_to_trade = int(trade_value / current_price)
                    
                    action = 'BUY' if shares_to_trade > 0 else 'SELL' if shares_to_trade < 0 else 'HOLD'
                    
                    allocation = PortfolioAllocation(
                        symbol=symbol,
                        target_weight=target_weight,
                        current_weight=current_weight,
                        shares_to_trade=abs(shares_to_trade),
                        trade_value=abs(trade_value),
                        action=action
                    )
                    
                    allocations.append(allocation)
                    
                except Exception as e:
                    logger.warning(f"Failed to process allocation for {symbol}: {e}")
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error generating allocation: {e}")
            return []

    def _needs_rebalancing(self, allocations: List[PortfolioAllocation]) -> bool:
        """Check if portfolio needs rebalancing"""
        for alloc in allocations:
            weight_difference = abs(alloc.target_weight - alloc.current_weight)
            if weight_difference > self.rebalance_threshold:
                return True
        return False

    async def _execute_rebalancing(self, allocations: List[PortfolioAllocation]):
        """Execute portfolio rebalancing (simulated)"""
        logger.info("Executing portfolio rebalancing...")
        
        for alloc in allocations:
            if alloc.action == 'HOLD':
                continue
                
            logger.info(f"{alloc.action} {alloc.shares_to_trade} shares of {alloc.symbol}")
            
            # Execute real trades
            if alloc.action == 'BUY':
                await self._execute_real_buy_order(alloc.symbol, alloc.shares_to_trade)
            elif alloc.action == 'SELL':
                await self._execute_real_sell_order(alloc.symbol, alloc.shares_to_trade)
        
        # Send rebalancing update
        await self._send_rebalancing_update(allocations)

    async def _execute_real_buy_order(self, symbol: str, shares: int):
        """Execute real buy order via orders API"""
        try:
            order_payload = {
                'ticker': symbol,
                'notional_usd': shares * await self._get_current_price(symbol),
                'last_price': await self._get_current_price(symbol),
                'side': 'buy'
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.orders_api_base}/orders",
                    json=order_payload,
                    headers={'Content-Type': 'application/json'}
                )

                if response.status_code in [200, 201]:
                    order = response.json()
                    logger.info(f"Real buy order placed for {symbol}: {order}")
                    return order
                else:
                    raise Exception(f"Order failed: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Real buy order failed for {symbol}: {e}")
            raise

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
            else:
                raise Exception(f"No price data for {symbol}")
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            raise

    async def _execute_real_sell_order(self, symbol: str, shares: int):
        """Execute real sell order via orders API"""
        try:
            if symbol not in self.positions:
                logger.warning(f"Cannot sell {symbol} - no position held")
                return

            pos = self.positions[symbol]
            shares_to_sell = min(shares, pos.shares)

            order_payload = {
                'ticker': symbol,
                'notional_usd': shares_to_sell * await self._get_current_price(symbol),
                'last_price': await self._get_current_price(symbol),
                'side': 'sell'
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.orders_api_base}/orders",
                    json=order_payload,
                    headers={'Content-Type': 'application/json'}
                )

                if response.status_code in [200, 201]:
                    order = response.json()
                    logger.info(f"Real sell order placed for {symbol}: {order}")
                    return order
                else:
                    raise Exception(f"Sell order failed: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Real sell order failed for {symbol}: {e}")
            raise

    async def _risk_monitoring_loop(self):
        """Monitor portfolio risk metrics"""
        while self.running:
            try:
                # Calculate risk metrics
                risk_metrics = await self._calculate_risk_metrics()
                
                # Check for risk violations
                violations = self._check_risk_violations(risk_metrics)
                
                if violations:
                    await self._handle_risk_violations(violations)
                
                # Store risk metrics
                await self._update_risk_metrics(risk_metrics)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(300)

    async def _calculate_risk_metrics(self) -> Optional[RiskMetrics]:
        """Calculate comprehensive risk metrics"""
        if not self.positions:
            return None
        
        try:
            # Get historical data for portfolio
            symbols = list(self.positions.keys())
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # Fetch data
            portfolio_data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    portfolio_data[symbol] = data['Close']
            
            if not portfolio_data:
                return None
            
            # Convert to DataFrame
            prices_df = pd.DataFrame(portfolio_data)
            returns_df = prices_df.pct_change().dropna()
            
            # Calculate portfolio returns
            weights = np.array([self.positions[symbol].weight for symbol in symbols])
            portfolio_returns = (returns_df * weights).sum(axis=1)
            
            # Risk metrics
            var_95, cvar_95 = self.risk_manager.calculate_var(portfolio_returns, 0.95)
            
            # Sharpe ratio (assuming 2% risk-free rate)
            excess_returns = portfolio_returns - 0.02/252
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
            
            # Volatility
            volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Max drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Beta and correlation with SPY
            spy_data = yf.Ticker('SPY').history(start=start_date, end=end_date)
            if not spy_data.empty:
                spy_returns = spy_data['Close'].pct_change().dropna()
                aligned_returns = portfolio_returns.align(spy_returns, join='inner')
                
                if len(aligned_returns[0]) > 30:
                    beta = np.cov(aligned_returns[0], aligned_returns[1])[0,1] / np.var(aligned_returns[1])
                    correlation = np.corrcoef(aligned_returns[0], aligned_returns[1])[0,1]
                else:
                    beta = 1.0
                    correlation = 0.5
            else:
                beta = 1.0
                correlation = 0.5
            
            return RiskMetrics(
                var_95=var_95 * 100,  # Convert to percentage
                cvar_95=cvar_95 * 100,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown * 100,
                volatility=volatility * 100,
                beta=beta,
                correlation_spy=correlation
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return None

    def _check_risk_violations(self, risk_metrics: RiskMetrics) -> List[str]:
        """Check for risk limit violations"""
        violations = []
        
        if risk_metrics.var_95 > 5.0:  # 5% daily VaR limit
            violations.append(f"Daily VaR exceeds limit: {risk_metrics.var_95:.2f}%")
        
        if risk_metrics.max_drawdown < -20.0:  # 20% max drawdown limit
            violations.append(f"Max drawdown exceeds limit: {risk_metrics.max_drawdown:.2f}%")
        
        if risk_metrics.volatility > 30.0:  # 30% annual volatility limit
            violations.append(f"Portfolio volatility too high: {risk_metrics.volatility:.2f}%")
        
        # Check concentration risk
        for symbol, position in self.positions.items():
            if position.weight > 0.20:  # 20% maximum position size
                violations.append(f"Position concentration risk in {symbol}: {position.weight:.1%}")
        
        return violations

    async def _handle_risk_violations(self, violations: List[str]):
        """Handle risk violations"""
        logger.warning(f"Risk violations detected: {violations}")
        
        # Send alert to master agent
        alert_data = {
            'type': 'alert',
            'sender': 'portfolio',
            'timestamp': datetime.now().isoformat(),
            'data': {
                'message': f"Risk violations: {', '.join(violations)}",
                'severity': 'high'
            }
        }
        
        self.redis_client.publish('master_channel', json.dumps(alert_data))

    async def _check_risk_limits(self):
        """Check various risk limits"""
        # Position size limits
        for symbol, position in self.positions.items():
            if position.weight > 0.15:  # 15% limit
                logger.warning(f"Position {symbol} exceeds weight limit: {position.weight:.1%}")
        
        # Daily loss limit
        if self.daily_pnl < -self.portfolio_value * 0.03:  # 3% daily loss limit
            logger.warning(f"Daily loss exceeds limit: {self.daily_pnl:.2f}")

    async def _update_portfolio_state(self):
        """Update Redis with current portfolio state"""
        try:
            portfolio_state = {
                'cash': self.cash,
                'portfolio_value': self.portfolio_value,
                'daily_pnl': self.daily_pnl,
                'positions': [asdict(pos) for pos in self.positions.values()],
                'timestamp': datetime.now().isoformat()
            }
            
            self.redis_client.set(
                'portfolio_state',
                json.dumps(portfolio_state, default=str),
                ex=300  # Expire in 5 minutes
            )
            
        except Exception as e:
            logger.error(f"Error updating portfolio state: {e}")

    async def _update_risk_metrics(self, risk_metrics: Optional[RiskMetrics]):
        """Update Redis with risk metrics"""
        if not risk_metrics:
            return
        
        try:
            self.redis_client.set(
                'portfolio_risk_metrics',
                json.dumps(asdict(risk_metrics), default=str),
                ex=3600  # Expire in 1 hour
            )
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")

    async def _send_rebalancing_update(self, allocations: List[PortfolioAllocation]):
        """Send rebalancing update to master agent"""
        try:
            update_data = {
                'type': 'rebalancing_completed',
                'sender': 'portfolio',
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'allocations': [asdict(alloc) for alloc in allocations],
                    'portfolio_value': self.portfolio_value,
                    'cash': self.cash,
                    'position_count': len(self.positions)
                }
            }
            
            self.redis_client.publish('master_channel', json.dumps(update_data))
            
        except Exception as e:
            logger.error(f"Error sending rebalancing update: {e}")

    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            try:
                self.redis_client.set('heartbeat:portfolio', datetime.now().isoformat())
                
                # Send status to master
                status_data = {
                    'type': 'status_update',
                    'sender': 'portfolio',
                    'data': {
                        'current_task': f'Managing {len(self.positions)} positions',
                        'metrics': {
                            'portfolio_value': self.portfolio_value,
                            'daily_pnl': self.daily_pnl,
                            'cash': self.cash,
                            'position_count': len(self.positions),
                            'largest_position': max([pos.weight for pos in self.positions.values()]) if self.positions else 0
                        }
                    }
                }
                
                self.redis_client.publish('master_channel', json.dumps(status_data))
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat: {e}")
                await asyncio.sleep(30)

async def main():
    """Main entry point"""
    portfolio_agent = PortfolioManagementAgent()
    
    try:
        await portfolio_agent.start()
    except Exception as e:
        logger.error(f"Fatal error in portfolio agent: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())