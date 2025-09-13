#!/usr/bin/env python3
"""
Backtesting Agent

Evaluates stock selection strategies on historical market data to assess 
their performance and risk characteristics.
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
import yfinance as yf
from anthropic import Anthropic
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BacktestingAgent')

@dataclass
class BacktestParameters:
    start_date: str
    end_date: str
    initial_capital: float
    max_positions: int
    stop_loss: float
    take_profit: float
    rebalance_frequency: str  # 'daily', 'weekly', 'monthly'
    commission: float = 0.001  # 0.1% commission

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

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name: str, parameters: Dict[str, Any]):
        self.name = name
        self.parameters = parameters
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate buy/sell signals. Override in subclasses."""
        raise NotImplementedError
    
    def filter_universe(self, universe: List[str]) -> List[str]:
        """Filter stock universe based on strategy criteria"""
        return universe

class MomentumStrategy(TradingStrategy):
    """Momentum-based trading strategy"""
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__('momentum', parameters)
        self.lookback = parameters.get('lookback', 20)
        self.threshold = parameters.get('threshold', 0.05)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum signals"""
        # Calculate momentum
        momentum = data['Close'].pct_change(self.lookback)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[momentum > self.threshold] = 1  # Buy signal
        signals[momentum < -self.threshold] = -1  # Sell signal
        
        return signals

class MeanReversionStrategy(TradingStrategy):
    """Mean reversion trading strategy"""
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__('mean_reversion', parameters)
        self.lookback = parameters.get('lookback', 20)
        self.threshold = parameters.get('threshold', 2.0)  # Standard deviations
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals"""
        # Calculate rolling mean and std
        rolling_mean = data['Close'].rolling(window=self.lookback).mean()
        rolling_std = data['Close'].rolling(window=self.lookback).std()
        
        # Calculate z-score
        z_score = (data['Close'] - rolling_mean) / rolling_std
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[z_score < -self.threshold] = 1  # Buy when oversold
        signals[z_score > self.threshold] = -1  # Sell when overbought
        
        return signals

class BreakoutStrategy(TradingStrategy):
    """Breakout trading strategy"""
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__('breakout', parameters)
        self.lookback = parameters.get('lookback', 20)
        self.volume_threshold = parameters.get('volume_threshold', 1.5)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate breakout signals"""
        # Calculate resistance/support levels
        rolling_high = data['High'].rolling(window=self.lookback).max()
        rolling_low = data['Low'].rolling(window=self.lookback).min()
        
        # Volume confirmation
        avg_volume = data['Volume'].rolling(window=self.lookback).mean()
        volume_surge = data['Volume'] / avg_volume
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Upward breakout with volume
        upward_breakout = (data['Close'] > rolling_high.shift(1)) & (volume_surge > self.volume_threshold)
        signals[upward_breakout] = 1
        
        # Downward breakdown with volume
        downward_breakout = (data['Close'] < rolling_low.shift(1)) & (volume_surge > self.volume_threshold)
        signals[downward_breakout] = -1
        
        return signals

class AISignalsStrategy(TradingStrategy):
    """AI-generated signals strategy"""
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__('ai_signals', parameters)
        self.min_score = parameters.get('min_score', 70)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate AI-based signals"""
        # This would integrate with the discovery agent's AI scores
        # For now, we'll simulate with technical indicators
        
        # Simple trend following with multiple indicators
        sma_short = data['Close'].rolling(window=10).mean()
        sma_long = data['Close'].rolling(window=30).mean()
        rsi = self._calculate_rsi(data['Close'], 14)
        
        signals = pd.Series(0, index=data.index)
        
        # Buy conditions: uptrend + oversold
        buy_condition = (sma_short > sma_long) & (rsi < 30)
        signals[buy_condition] = 1
        
        # Sell conditions: downtrend + overbought
        sell_condition = (sma_short < sma_long) & (rsi > 70)
        signals[sell_condition] = -1
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

class BacktestingAgent:
    def __init__(self):
        self.claude = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
        self.redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
        self.running = False
        
        # Strategy registry
        self.strategies = {
            'momentum': MomentumStrategy,
            'mean_reversion': MeanReversionStrategy,
            'breakout': BreakoutStrategy,
            'ai_signals': AISignalsStrategy
        }
        
        # Results storage
        self.backtest_results: List[BacktestResult] = []
        
        # Default stock universe
        self.stock_universe = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'DIS', 'PYPL', 'ADBE', 'CRM', 'INTC', 'AMD', 'UBER', 'SPOT'
        ]

    async def start(self):
        """Start the backtesting agent"""
        logger.info("Starting Backtesting Agent...")
        self.running = True
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._optimization_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down Backtesting Agent...")
            self.running = False

    async def run_backtest(self, strategy_name: str, parameters: Dict[str, Any]) -> BacktestResult:
        """Run a comprehensive backtest"""
        logger.info(f"Running backtest for {strategy_name} strategy...")
        
        try:
            # Parse parameters
            backtest_params = BacktestParameters(
                start_date=parameters.get('startDate', '2024-01-01'),
                end_date=parameters.get('endDate', '2024-12-31'),
                initial_capital=parameters.get('initialCapital', 100000),
                max_positions=parameters.get('maxPositions', 10),
                stop_loss=parameters.get('stopLoss', 0.05),
                take_profit=parameters.get('takeProfit', 0.15),
                rebalance_frequency=parameters.get('rebalanceFrequency', 'weekly'),
                commission=parameters.get('commission', 0.001)
            )
            
            # Create strategy instance
            if strategy_name not in self.strategies:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            
            strategy_class = self.strategies[strategy_name]
            strategy = strategy_class(parameters)
            
            # Run backtest
            result = await self._execute_backtest(strategy, backtest_params)
            
            # Store result
            self.backtest_results.insert(0, result)
            
            # Update Redis
            await self._update_results()
            
            # Send update to master agent
            await self._send_backtest_update(result)
            
            logger.info(f"Backtest completed for {strategy_name}: {result.total_return:.2f}% return")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise

    async def _execute_backtest(self, strategy: TradingStrategy, params: BacktestParameters) -> BacktestResult:
        """Execute the actual backtest simulation"""
        
        # Fetch historical data
        data = await self._fetch_historical_data(params.start_date, params.end_date)
        
        if data.empty:
            raise ValueError("No historical data available for backtesting")
        
        # Initialize portfolio
        portfolio = {
            'cash': params.initial_capital,
            'positions': {},
            'equity': [params.initial_capital],
            'dates': [pd.to_datetime(params.start_date)]
        }
        
        trade_log = []
        
        # Simulate trading
        for date in pd.date_range(start=params.start_date, end=params.end_date, freq='D'):
            if date.weekday() >= 5:  # Skip weekends
                continue
                
            date_str = date.strftime('%Y-%m-%d')
            
            # Get data up to current date
            current_data = data[data.index <= date]
            if len(current_data) < 30:  # Need minimum data
                continue
            
            # Rebalance portfolio (weekly/monthly)
            if self._should_rebalance(date, params.rebalance_frequency):
                await self._rebalance_portfolio(
                    portfolio, strategy, current_data, params, trade_log, date
                )
            
            # Update portfolio value
            portfolio_value = await self._calculate_portfolio_value(
                portfolio, data, date
            )
            
            portfolio['equity'].append(portfolio_value)
            portfolio['dates'].append(date)
        
        # Calculate performance metrics
        return self._calculate_performance_metrics(
            strategy.name, params, portfolio, trade_log
        )

    async def _fetch_historical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data for backtesting"""
        try:
            # For demo, we'll use a representative stock (SPY)
            ticker = yf.Ticker('SPY')
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                # Fallback to mock data
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                data = pd.DataFrame(index=date_range)
                
                # Generate synthetic price data
                np.random.seed(42)
                base_price = 400.0
                returns = np.random.normal(0.0005, 0.02, len(date_range))
                prices = [base_price]
                
                for ret in returns:
                    prices.append(prices[-1] * (1 + ret))
                
                data['Close'] = prices[1:]
                data['Open'] = data['Close'] * (1 + np.random.normal(0, 0.005, len(data)))
                data['High'] = data[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, len(data))))
                data['Low'] = data[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, len(data))))
                data['Volume'] = np.random.randint(50000000, 200000000, len(data))
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def _should_rebalance(self, date: pd.Timestamp, frequency: str) -> bool:
        """Check if portfolio should be rebalanced"""
        if frequency == 'daily':
            return True
        elif frequency == 'weekly':
            return date.weekday() == 0  # Monday
        elif frequency == 'monthly':
            return date.day == 1
        return False

    async def _rebalance_portfolio(self, portfolio: Dict[str, Any], strategy: TradingStrategy,
                                 data: pd.DataFrame, params: BacktestParameters,
                                 trade_log: List[Dict[str, Any]], current_date: pd.Timestamp):
        """Rebalance portfolio based on strategy signals"""
        
        # Generate signals for current date
        signals = strategy.generate_signals(data)
        current_signal = signals.iloc[-1] if len(signals) > 0 else 0
        
        current_price = data['Close'].iloc[-1]
        
        # Simple position management
        if current_signal == 1 and len(portfolio['positions']) < params.max_positions:
            # Buy signal - allocate equal weight
            position_size = portfolio['cash'] / (params.max_positions - len(portfolio['positions']))
            shares = int(position_size / current_price)
            
            if shares > 0:
                cost = shares * current_price * (1 + params.commission)
                if cost <= portfolio['cash']:
                    portfolio['cash'] -= cost
                    portfolio['positions']['SPY'] = {
                        'shares': shares,
                        'avg_price': current_price,
                        'entry_date': current_date
                    }
                    
                    trade_log.append({
                        'date': current_date,
                        'symbol': 'SPY',
                        'action': 'BUY',
                        'shares': shares,
                        'price': current_price,
                        'value': cost
                    })
        
        elif current_signal == -1 and 'SPY' in portfolio['positions']:
            # Sell signal
            position = portfolio['positions']['SPY']
            shares = position['shares']
            proceeds = shares * current_price * (1 - params.commission)
            
            portfolio['cash'] += proceeds
            del portfolio['positions']['SPY']
            
            trade_log.append({
                'date': current_date,
                'symbol': 'SPY',
                'action': 'SELL',
                'shares': shares,
                'price': current_price,
                'value': proceeds
            })
        
        # Check stop loss/take profit
        if 'SPY' in portfolio['positions']:
            position = portfolio['positions']['SPY']
            entry_price = position['avg_price']
            current_return = (current_price - entry_price) / entry_price
            
            if current_return <= -params.stop_loss or current_return >= params.take_profit:
                # Exit position
                shares = position['shares']
                proceeds = shares * current_price * (1 - params.commission)
                
                portfolio['cash'] += proceeds
                del portfolio['positions']['SPY']
                
                trade_log.append({
                    'date': current_date,
                    'symbol': 'SPY',
                    'action': 'SELL' if current_return <= -params.stop_loss else 'PROFIT',
                    'shares': shares,
                    'price': current_price,
                    'value': proceeds
                })

    async def _calculate_portfolio_value(self, portfolio: Dict[str, Any], 
                                       data: pd.DataFrame, date: pd.Timestamp) -> float:
        """Calculate total portfolio value"""
        total_value = portfolio['cash']
        
        # Add position values
        if 'SPY' in portfolio['positions'] and date in data.index:
            position = portfolio['positions']['SPY']
            current_price = data.loc[date, 'Close']
            total_value += position['shares'] * current_price
        
        return total_value

    def _calculate_performance_metrics(self, strategy_name: str, params: BacktestParameters,
                                     portfolio: Dict[str, Any], trade_log: List[Dict[str, Any]]) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        
        equity_curve = portfolio['equity']
        dates = portfolio['dates']
        
        # Convert to pandas series for calculations
        equity_series = pd.Series(equity_curve, index=dates)
        returns = equity_series.pct_change().dropna()
        
        # Calculate metrics
        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        
        # Annualized return
        days = (dates[-1] - dates[0]).days
        annual_return = ((equity_curve[-1] / equity_curve[0]) ** (365.25 / days) - 1) * 100
        
        # Volatility
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming 2% risk-free rate)
        excess_returns = returns - 0.02/252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Max drawdown
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        # Trade statistics
        total_trades = len(trade_log)
        if total_trades > 0:
            trade_returns = []
            for trade in trade_log:
                if trade['action'] in ['SELL', 'PROFIT']:
                    # Find corresponding buy trade
                    buy_trades = [t for t in trade_log if t['action'] == 'BUY' and t['symbol'] == trade['symbol']]
                    if buy_trades:
                        buy_price = buy_trades[-1]['price']  # Most recent buy
                        trade_return = (trade['price'] - buy_price) / buy_price
                        trade_returns.append(trade_return)
            
            win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns) * 100 if trade_returns else 0
            avg_trade_return = np.mean(trade_returns) * 100 if trade_returns else 0
            best_trade = max(trade_returns) * 100 if trade_returns else 0
            worst_trade = min(trade_returns) * 100 if trade_returns else 0
        else:
            win_rate = 0
            avg_trade_return = 0
            best_trade = 0
            worst_trade = 0
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return - 2) / (downside_deviation * 100) if downside_deviation > 0 else 0
        
        # Monthly returns
        monthly_returns = equity_series.resample('M').last().pct_change().dropna() * 100
        
        # Equity curve for visualization
        equity_curve_data = [
            {'date': date.isoformat(), 'value': value}
            for date, value in zip(dates, equity_curve)
        ]
        
        return BacktestResult(
            strategy=strategy_name,
            parameters=params,
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            avg_trade_return=avg_trade_return,
            best_trade=best_trade,
            worst_trade=worst_trade,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            period=f"{params.start_date} to {params.end_date}",
            equity_curve=equity_curve_data,
            trade_log=trade_log,
            monthly_returns=monthly_returns.tolist()
        )

    async def _update_results(self):
        """Update Redis with backtest results"""
        try:
            results_data = [asdict(result) for result in self.backtest_results]
            
            # Handle datetime serialization
            def serialize_datetime(obj):
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                return str(obj)
            
            self.redis_client.set(
                'backtest_results',
                json.dumps(results_data, default=serialize_datetime),
                ex=3600  # Expire in 1 hour
            )
            
        except Exception as e:
            logger.error(f"Error updating results in Redis: {e}")

    async def _send_backtest_update(self, result: BacktestResult):
        """Send backtest completion update to master agent"""
        try:
            update_data = {
                'type': 'backtest_completed',
                'sender': 'backtesting',
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'strategy': result.strategy,
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'total_trades': result.total_trades
                }
            }
            
            self.redis_client.publish('master_channel', json.dumps(update_data))
            
        except Exception as e:
            logger.error(f"Error sending backtest update: {e}")

    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            try:
                self.redis_client.set('heartbeat:backtesting', datetime.now().isoformat())
                
                # Send status to master
                status_data = {
                    'type': 'status_update',
                    'sender': 'backtesting',
                    'data': {
                        'current_task': 'Ready for backtesting requests',
                        'metrics': {
                            'completed_backtests': len(self.backtest_results),
                            'avg_return': np.mean([r.total_return for r in self.backtest_results]) if self.backtest_results else 0,
                            'best_strategy': max(self.backtest_results, key=lambda x: x.total_return).strategy if self.backtest_results else 'None'
                        }
                    }
                }
                
                self.redis_client.publish('master_channel', json.dumps(status_data))
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat: {e}")
                await asyncio.sleep(30)

    async def _optimization_loop(self):
        """Periodically run strategy optimization"""
        while self.running:
            try:
                # Run optimization every hour
                await asyncio.sleep(3600)
                
                logger.info("Running strategy optimization...")
                
                # Use Claude to suggest parameter optimizations
                await self._optimize_strategies()
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(3600)

    async def _optimize_strategies(self):
        """Use Claude to suggest strategy optimizations"""
        try:
            if not self.backtest_results:
                return
            
            # Prepare performance data for Claude
            performance_summary = []
            for result in self.backtest_results[-10:]:  # Last 10 results
                performance_summary.append({
                    'strategy': result.strategy,
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'parameters': asdict(result.parameters)
                })
            
            prompt = f"""
            Analyze these backtesting results and suggest optimizations:
            
            {json.dumps(performance_summary, indent=2, default=str)}
            
            Please provide:
            1. Which strategies are performing best/worst
            2. Parameter adjustments that might improve performance
            3. New strategy ideas based on the results
            4. Risk management improvements
            
            Focus on actionable recommendations for improving risk-adjusted returns.
            """
            
            response = self.claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            optimization_suggestions = response.content[0].text
            
            # Store suggestions in Redis
            self.redis_client.set(
                'optimization_suggestions',
                optimization_suggestions,
                ex=7200  # Expire in 2 hours
            )
            
            logger.info("Strategy optimization completed")
            
        except Exception as e:
            logger.error(f"Error in strategy optimization: {e}")

async def main():
    """Main entry point"""
    backtesting_agent = BacktestingAgent()
    
    try:
        await backtesting_agent.start()
    except Exception as e:
        logger.error(f"Fatal error in backtesting agent: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())