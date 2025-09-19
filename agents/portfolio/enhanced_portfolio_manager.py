#!/usr/bin/env python3
"""
Enhanced Portfolio Management System

Comprehensive portfolio monitoring and management system that:
1. Tracks all positions from the discovery system
2. Continuously evaluates stock health using multiple criteria
3. Provides AI-powered buy/sell/hold recommendations
4. Manages risk and position sizing
5. Delivers explosive returns through systematic management
"""

import asyncio
import logging
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
import yfinance as yf
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EnhancedPortfolioManager')

@dataclass
class PortfolioPosition:
    """Enhanced position tracking with health metrics"""
    symbol: str
    shares: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    weight: float
    entry_date: datetime
    
    # Health metrics
    technical_health: float  # 0-100 score
    fundamental_health: float  # 0-100 score
    thesis_health: float  # 0-100 score
    overall_health: float  # 0-100 composite score
    
    # Discovery system data
    original_score: float
    discovery_date: datetime
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    
    # Performance tracking
    days_held: int = 0
    peak_price: float = 0.0
    max_drawdown: float = 0.0
    thesis_performance: str = "ON_TRACK"  # ON_TRACK, AHEAD, BEHIND, FAILED

@dataclass
class PositionRecommendation:
    """AI-powered position recommendation"""
    symbol: str
    action: str  # HOLD, ADD, TRIM, EXIT
    confidence: float  # 0-100
    rationale: str
    urgency: str  # LOW, MEDIUM, HIGH, CRITICAL
    suggested_shares: Optional[int] = None
    target_weight: Optional[float] = None
    risk_factors: List[str] = None

@dataclass
class PortfolioHealth:
    """Overall portfolio health metrics"""
    total_value: float
    daily_pnl: float
    daily_pnl_percent: float
    total_pnl: float
    total_pnl_percent: float
    
    # Risk metrics
    concentration_risk: float  # 0-100 (higher = more concentrated)
    sector_diversification: float  # 0-100 (higher = more diversified)
    volatility_score: float  # 0-100 (higher = more volatile)
    correlation_risk: float  # 0-100 (higher = more correlated)
    
    # Performance metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    avg_winner: float
    avg_loser: float
    
    # Health scores
    technical_health: float  # Portfolio-wide technical health
    fundamental_health: float  # Portfolio-wide fundamental health
    overall_health: float  # Composite health score

class AlpacaPortfolioClient:
    """Enhanced Alpaca API client for portfolio management"""
    
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not found in environment")
        
        self.api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.api_secret,
            base_url=self.base_url,
            api_version='v2'
        )
        
        # Verify connection
        try:
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca account: {account.id}")
            logger.info(f"Account status: {account.status}")
            logger.info(f"Buying power: ${float(account.buying_power):,.2f}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise
    
    def get_positions(self) -> List[Dict]:
        """Get current portfolio positions"""
        try:
            positions = self.api.list_positions()
            return [{
                'symbol': pos.symbol,
                'shares': float(pos.qty),
                'avg_price': float(pos.avg_cost_basis),
                'current_price': float(pos.current_price) if pos.current_price else 0.0,
                'market_value': float(pos.market_value) if pos.market_value else 0.0,
                'unrealized_pnl': float(pos.unrealized_pl) if pos.unrealized_pl else 0.0,
                'unrealized_pnl_percent': float(pos.unrealized_plpc) if pos.unrealized_plpc else 0.0
            } for pos in positions]
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'day_trade_buying_power': float(account.day_trade_buying_power),
                'status': account.status
            }
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return {}
    
    def get_portfolio_history(self, period: str = '1M') -> Dict:
        """Get portfolio performance history"""
        try:
            history = self.api.get_portfolio_history(
                period=period,
                timeframe='1D'
            )
            return {
                'timestamps': history.timestamp,
                'equity': history.equity,
                'profit_loss': history.profit_loss,
                'profit_loss_pct': history.profit_loss_pct
            }
        except Exception as e:
            logger.error(f"Error fetching portfolio history: {e}")
            return {}

class StockHealthAnalyzer:
    """Analyzes individual stock health using multiple criteria"""
    
    def __init__(self):
        self.claude = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
    
    def analyze_technical_health(self, symbol: str, position: PortfolioPosition) -> float:
        """Analyze technical health (0-100)"""
        try:
            # Get recent price data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            
            if hist.empty:
                return 50.0  # Neutral if no data
            
            current_price = position.current_price
            scores = []
            
            # 1. Price vs Moving Averages (25%)
            if len(hist) >= 20:
                ma_20 = hist['Close'].rolling(20).mean().iloc[-1]
                ma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else ma_20
                
                if current_price > ma_20 > ma_50:
                    ma_score = 100  # Strong uptrend
                elif current_price > ma_20:
                    ma_score = 75   # Above short-term MA
                elif current_price > ma_50:
                    ma_score = 50   # Above long-term MA
                else:
                    ma_score = 25   # Below both MAs
                
                scores.append(('moving_averages', ma_score, 0.25))
            
            # 2. Volume Trend (20%)
            if len(hist) >= 10:
                recent_volume = hist['Volume'].tail(5).mean()
                avg_volume = hist['Volume'].mean()
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
                
                if volume_ratio >= 2.0:
                    volume_score = 100  # High volume surge
                elif volume_ratio >= 1.5:
                    volume_score = 80   # Good volume
                elif volume_ratio >= 1.0:
                    volume_score = 60   # Average volume
                else:
                    volume_score = 30   # Low volume
                
                scores.append(('volume_trend', volume_score, 0.20))
            
            # 3. Momentum (25%)
            returns_5d = (current_price / hist['Close'].iloc[-6] - 1) * 100 if len(hist) >= 6 else 0
            returns_20d = (current_price / hist['Close'].iloc[-21] - 1) * 100 if len(hist) >= 21 else 0
            
            if returns_5d > 5 and returns_20d > 10:
                momentum_score = 100  # Strong momentum
            elif returns_5d > 0 and returns_20d > 0:
                momentum_score = 75   # Positive momentum
            elif returns_5d > 0 or returns_20d > 0:
                momentum_score = 50   # Mixed momentum
            else:
                momentum_score = 25   # Negative momentum
            
            scores.append(('momentum', momentum_score, 0.25))
            
            # 4. Volatility Health (15%)
            if len(hist) >= 20:
                volatility = hist['Close'].pct_change().std() * np.sqrt(252)
                
                if 0.15 <= volatility <= 0.35:  # Optimal volatility range
                    vol_score = 100
                elif 0.10 <= volatility <= 0.50:
                    vol_score = 75
                elif volatility <= 0.70:
                    vol_score = 50
                else:
                    vol_score = 25  # Too volatile
                
                scores.append(('volatility', vol_score, 0.15))
            
            # 5. Support/Resistance (15%)
            if len(hist) >= 20:
                recent_high = hist['High'].tail(20).max()
                recent_low = hist['Low'].tail(20).min()
                
                price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
                
                if price_position >= 0.8:
                    sr_score = 100  # Near resistance (bullish)
                elif price_position >= 0.6:
                    sr_score = 75   # Upper range
                elif price_position >= 0.4:
                    sr_score = 50   # Middle range
                elif price_position >= 0.2:
                    sr_score = 25   # Lower range
                else:
                    sr_score = 10   # Near support (bearish)
                
                scores.append(('support_resistance', sr_score, 0.15))
            
            # Calculate weighted technical health score
            if scores:
                technical_health = sum(score * weight for _, score, weight in scores)
                return min(max(technical_health, 0), 100)
            
            return 50.0
            
        except Exception as e:
            logger.warning(f"Error analyzing technical health for {symbol}: {e}")
            return 50.0
    
    def analyze_fundamental_health(self, symbol: str, position: PortfolioPosition) -> float:
        """Analyze fundamental health (0-100)"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            scores = []
            
            # 1. Financial Strength (40%)
            financial_score = 60  # Default score
            
            # Revenue growth
            if 'revenueGrowth' in info and info['revenueGrowth']:
                growth = info['revenueGrowth']
                if growth > 0.20:  # 20%+ growth
                    financial_score += 20
                elif growth > 0.10:  # 10%+ growth
                    financial_score += 10
                elif growth > 0:
                    financial_score += 5
            
            # Profit margins
            if 'profitMargins' in info and info['profitMargins']:
                margin = info['profitMargins']
                if margin > 0.15:  # 15%+ margin
                    financial_score += 15
                elif margin > 0.05:  # 5%+ margin
                    financial_score += 5
            
            # Debt to equity
            if 'debtToEquity' in info and info['debtToEquity']:
                debt_ratio = info['debtToEquity']
                if debt_ratio < 0.3:  # Low debt
                    financial_score += 5
                elif debt_ratio > 1.0:  # High debt
                    financial_score -= 10
            
            scores.append(('financial_strength', min(financial_score, 100), 0.40))
            
            # 2. Valuation (30%)
            valuation_score = 50  # Default neutral
            
            # P/E ratio analysis
            if 'trailingPE' in info and info['trailingPE']:
                pe = info['trailingPE']
                if pe < 15:  # Undervalued
                    valuation_score = 80
                elif pe < 25:  # Fair value
                    valuation_score = 60
                elif pe < 35:  # Slightly overvalued
                    valuation_score = 40
                else:  # Overvalued
                    valuation_score = 20
            
            scores.append(('valuation', valuation_score, 0.30))
            
            # 3. Market Position (30%)
            market_score = 60  # Default
            
            # Market cap
            if 'marketCap' in info and info['marketCap']:
                market_cap = info['marketCap']
                if 1e9 <= market_cap <= 50e9:  # Sweet spot for explosive moves
                    market_score += 20
                elif market_cap >= 50e9:  # Large cap
                    market_score += 10
            
            # Sector momentum (simplified)
            sector = info.get('sector', 'Unknown')
            if sector in ['Technology', 'Healthcare', 'Consumer Discretionary']:
                market_score += 10  # Growth sectors
            
            scores.append(('market_position', min(market_score, 100), 0.30))
            
            # Calculate weighted fundamental health
            if scores:
                fundamental_health = sum(score * weight for _, score, weight in scores)
                return min(max(fundamental_health, 0), 100)
            
            return 50.0
            
        except Exception as e:
            logger.warning(f"Error analyzing fundamental health for {symbol}: {e}")
            return 50.0
    
    def analyze_thesis_health(self, symbol: str, position: PortfolioPosition) -> Tuple[float, str]:
        """Analyze how well the stock is performing vs. original thesis"""
        try:
            days_held = position.days_held
            current_return = position.unrealized_pnl_percent
            target_return = ((position.price_target / position.avg_price) - 1) * 100 if position.price_target else 63.8  # Default explosive target
            
            # Expected timeline for explosive moves (typically 1-3 months)
            expected_days = 60  # 2 months average
            
            # Calculate expected return based on time held
            expected_return_now = (days_held / expected_days) * target_return
            
            # Performance vs. expectation
            performance_ratio = current_return / expected_return_now if expected_return_now != 0 else 1.0
            
            if performance_ratio >= 1.5:  # 50% ahead of schedule
                thesis_health = 100
                thesis_status = "AHEAD"
            elif performance_ratio >= 1.0:  # On track
                thesis_health = 85
                thesis_status = "ON_TRACK"
            elif performance_ratio >= 0.5:  # Slightly behind
                thesis_health = 60
                thesis_status = "BEHIND"
            elif performance_ratio >= 0:  # Significantly behind but positive
                thesis_health = 40
                thesis_status = "BEHIND"
            else:  # Losing money
                thesis_health = 20
                thesis_status = "FAILED"
            
            # Adjust for time factor
            if days_held > expected_days * 1.5:  # Held too long
                thesis_health *= 0.8
                if thesis_status == "ON_TRACK":
                    thesis_status = "BEHIND"
            
            return min(max(thesis_health, 0), 100), thesis_status
            
        except Exception as e:
            logger.warning(f"Error analyzing thesis health for {symbol}: {e}")
            return 50.0, "UNKNOWN"
    
    async def generate_recommendation(self, position: PortfolioPosition) -> PositionRecommendation:
        """Generate AI-powered recommendation for position"""
        try:
            # Prepare position data for Claude
            position_data = {
                'symbol': position.symbol,
                'current_price': position.current_price,
                'avg_price': position.avg_price,
                'unrealized_pnl_percent': position.unrealized_pnl_percent,
                'days_held': position.days_held,
                'weight': position.weight,
                'technical_health': position.technical_health,
                'fundamental_health': position.fundamental_health,
                'thesis_health': position.thesis_health,
                'overall_health': position.overall_health,
                'thesis_performance': position.thesis_performance,
                'price_target': position.price_target,
                'stop_loss': position.stop_loss,
                'max_drawdown': position.max_drawdown
            }
            
            prompt = f"""
As a hedge fund portfolio manager focused on explosive stock returns, analyze this position and provide a recommendation.

Position Data:
{json.dumps(position_data, indent=2)}

Context:
- Our strategy targets explosive moves (50-100%+ returns) within 1-3 months
- We achieved 63.8% average returns in previous periods with stocks like VIGL (+324%), CRWV (+171%), AEVA (+162%)
- Health scores are 0-100 (higher is better)
- We use systematic risk management with stop losses

Provide a recommendation in this exact JSON format:
{{
    "action": "HOLD|ADD|TRIM|EXIT",
    "confidence": 85,
    "rationale": "Clear explanation of reasoning",
    "urgency": "LOW|MEDIUM|HIGH|CRITICAL",
    "suggested_shares": 100,
    "target_weight": 0.08,
    "risk_factors": ["Risk factor 1", "Risk factor 2"]
}}

Action Guidelines:
- HOLD: Stock maintaining thesis, continue current position
- ADD: Opportunity to increase position (dip buying, breakout confirmation)
- TRIM: Take partial profits or reduce overweight position
- EXIT: Stop loss triggered, thesis failed, or better opportunities

Focus on explosive return potential while managing downside risk.
"""
            
            response = self.claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse Claude's response
            response_text = response.content[0].text
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                recommendation_data = json.loads(json_match.group())
                
                return PositionRecommendation(
                    symbol=position.symbol,
                    action=recommendation_data.get('action', 'HOLD'),
                    confidence=recommendation_data.get('confidence', 50),
                    rationale=recommendation_data.get('rationale', 'AI analysis completed'),
                    urgency=recommendation_data.get('urgency', 'MEDIUM'),
                    suggested_shares=recommendation_data.get('suggested_shares'),
                    target_weight=recommendation_data.get('target_weight'),
                    risk_factors=recommendation_data.get('risk_factors', [])
                )
            else:
                # Fallback recommendation
                return self._generate_fallback_recommendation(position)
            
        except Exception as e:
            logger.error(f"Error generating recommendation for {position.symbol}: {e}")
            return self._generate_fallback_recommendation(position)
    
    def _generate_fallback_recommendation(self, position: PortfolioPosition) -> PositionRecommendation:
        """Generate fallback recommendation based on simple rules"""
        # Simple rule-based recommendation
        if position.overall_health >= 80 and position.unrealized_pnl_percent > 50:
            action = "TRIM"
            rationale = "Strong performance, consider taking profits"
            urgency = "MEDIUM"
        elif position.overall_health >= 70:
            action = "HOLD"
            rationale = "Healthy position, maintain current allocation"
            urgency = "LOW"
        elif position.overall_health >= 50:
            action = "HOLD"
            rationale = "Average health, monitor closely"
            urgency = "MEDIUM"
        elif position.unrealized_pnl_percent < -15:
            action = "EXIT"
            rationale = "Stop loss triggered, preserve capital"
            urgency = "HIGH"
        else:
            action = "HOLD"
            rationale = "Below average health, consider exit if deteriorates"
            urgency = "HIGH"
        
        return PositionRecommendation(
            symbol=position.symbol,
            action=action,
            confidence=60,
            rationale=rationale,
            urgency=urgency,
            risk_factors=["Market volatility", "Position health below optimal"]
        )

class EnhancedPortfolioManager:
    """Main portfolio management system"""
    
    def __init__(self):
        self.alpaca_client = AlpacaPortfolioClient()
        self.health_analyzer = StockHealthAnalyzer()
        
        # Portfolio state
        self.positions: Dict[str, PortfolioPosition] = {}
        self.recommendations: Dict[str, PositionRecommendation] = {}
        self.portfolio_health: Optional[PortfolioHealth] = None
        
        # Configuration
        self.discovery_api_url = os.getenv('VITE_DISCOVERY_API_URL', 'https://alphastack-discovery.onrender.com')
        self.update_interval = 300  # 5 minutes
        
        self.running = False
        
        logger.info("Enhanced Portfolio Manager initialized")
    
    async def start(self):
        """Start the portfolio management system"""
        logger.info("Starting Enhanced Portfolio Management System...")
        self.running = True
        
        # Initial portfolio load
        await self.update_portfolio()
        
        # Start monitoring loops
        tasks = [
            asyncio.create_task(self._portfolio_monitoring_loop()),
            asyncio.create_task(self._health_analysis_loop()),
            asyncio.create_task(self._recommendation_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down Enhanced Portfolio Manager...")
            self.running = False
    
    async def update_portfolio(self):
        """Update portfolio positions and health metrics"""
        try:
            # Get current positions from Alpaca
            alpaca_positions = self.alpaca_client.get_positions()
            account_info = self.alpaca_client.get_account_info()
            
            if not alpaca_positions:
                logger.info("No positions found in Alpaca account")
                return
            
            # Update position objects
            for pos_data in alpaca_positions:
                symbol = pos_data['symbol']
                
                # Calculate additional metrics
                entry_date = self.positions[symbol].entry_date if symbol in self.positions else datetime.now()
                days_held = (datetime.now() - entry_date).days
                
                # Get discovery data for this stock
                discovery_data = await self._get_discovery_data(symbol)
                
                position = PortfolioPosition(
                    symbol=symbol,
                    shares=pos_data['shares'],
                    avg_price=pos_data['avg_price'],
                    current_price=pos_data['current_price'],
                    market_value=pos_data['market_value'],
                    unrealized_pnl=pos_data['unrealized_pnl'],
                    unrealized_pnl_percent=pos_data['unrealized_pnl_percent'],
                    weight=pos_data['market_value'] / account_info.get('portfolio_value', 1),
                    entry_date=entry_date,
                    
                    # Initialize health metrics (will be updated by analyzer)
                    technical_health=50.0,
                    fundamental_health=50.0,
                    thesis_health=50.0,
                    overall_health=50.0,
                    
                    # Discovery system data
                    original_score=discovery_data.get('score', 50),
                    discovery_date=discovery_data.get('discovery_date', entry_date),
                    price_target=discovery_data.get('price_target'),
                    stop_loss=discovery_data.get('stop_loss'),
                    
                    # Performance tracking
                    days_held=days_held,
                    peak_price=max(self.positions[symbol].peak_price if symbol in self.positions else 0, pos_data['current_price']),
                    max_drawdown=self._calculate_max_drawdown(symbol, pos_data['current_price']),
                    thesis_performance="ON_TRACK"
                )
                
                self.positions[symbol] = position
            
            # Calculate portfolio health
            self.portfolio_health = self._calculate_portfolio_health()
            
            logger.info(f"Portfolio updated: {len(self.positions)} positions, ${account_info.get('portfolio_value', 0):,.2f} total value")
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
    
    async def _get_discovery_data(self, symbol: str) -> Dict:
        """Get discovery data for a symbol"""
        try:
            response = requests.get(f"{self.discovery_api_url}/signals/top", timeout=10)
            if response.status_code == 200:
                data = response.json()
                signals = data.get('signals', data.get('final_recommendations', []))
                
                for signal in signals:
                    if signal.get('symbol') == symbol:
                        return {
                            'score': signal.get('score', signal.get('accumulation_score', 50)),
                            'price_target': signal.get('price_target'),
                            'stop_loss': signal.get('stop_loss'),
                            'discovery_date': datetime.now()  # Simplified
                        }
            return {}
        except Exception as e:
            logger.warning(f"Could not fetch discovery data for {symbol}: {e}")
            return {}
    
    def _calculate_max_drawdown(self, symbol: str, current_price: float) -> float:
        """Calculate maximum drawdown for position"""
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        peak_price = max(position.peak_price, current_price)
        drawdown = (peak_price - current_price) / peak_price * 100 if peak_price > 0 else 0.0
        return max(position.max_drawdown, drawdown)
    
    def _calculate_portfolio_health(self) -> PortfolioHealth:
        """Calculate overall portfolio health metrics"""
        if not self.positions:
            return PortfolioHealth(
                total_value=0, daily_pnl=0, daily_pnl_percent=0,
                total_pnl=0, total_pnl_percent=0,
                concentration_risk=0, sector_diversification=100,
                volatility_score=50, correlation_risk=50,
                sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
                win_rate=0, avg_winner=0, avg_loser=0,
                technical_health=50, fundamental_health=50, overall_health=50
            )
        
        # Basic portfolio metrics
        total_value = sum(pos.market_value for pos in self.positions.values())
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_cost = sum(pos.shares * pos.avg_price for pos in self.positions.values())
        total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        # Risk metrics
        weights = [pos.weight for pos in self.positions.values()]
        concentration_risk = max(weights) * 100 if weights else 0  # Highest single position weight
        
        # Health scores
        technical_health = np.mean([pos.technical_health for pos in self.positions.values()])
        fundamental_health = np.mean([pos.fundamental_health for pos in self.positions.values()])
        overall_health = np.mean([pos.overall_health for pos in self.positions.values()])
        
        # Win/loss statistics
        winners = [pos for pos in self.positions.values() if pos.unrealized_pnl > 0]
        losers = [pos for pos in self.positions.values() if pos.unrealized_pnl < 0]
        
        win_rate = len(winners) / len(self.positions) * 100 if self.positions else 0
        avg_winner = np.mean([pos.unrealized_pnl_percent for pos in winners]) if winners else 0
        avg_loser = np.mean([pos.unrealized_pnl_percent for pos in losers]) if losers else 0
        
        return PortfolioHealth(
            total_value=total_value,
            daily_pnl=0,  # Would need historical data
            daily_pnl_percent=0,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            
            concentration_risk=concentration_risk,
            sector_diversification=min(100, len(self.positions) * 10),  # Simplified
            volatility_score=50,  # Would need historical calculation
            correlation_risk=50,  # Would need correlation matrix
            
            sharpe_ratio=0,  # Would need return history
            sortino_ratio=0,
            max_drawdown=max([pos.max_drawdown for pos in self.positions.values()], default=0),
            
            win_rate=win_rate,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            
            technical_health=technical_health,
            fundamental_health=fundamental_health,
            overall_health=overall_health
        )
    
    async def _portfolio_monitoring_loop(self):
        """Main portfolio monitoring loop"""
        while self.running:
            try:
                await self.update_portfolio()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in portfolio monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _health_analysis_loop(self):
        """Analyze health of all positions"""
        while self.running:
            try:
                for symbol, position in self.positions.items():
                    # Analyze technical health
                    technical_health = self.health_analyzer.analyze_technical_health(symbol, position)
                    
                    # Analyze fundamental health
                    fundamental_health = self.health_analyzer.analyze_fundamental_health(symbol, position)
                    
                    # Analyze thesis health
                    thesis_health, thesis_status = self.health_analyzer.analyze_thesis_health(symbol, position)
                    
                    # Calculate overall health (weighted average)
                    overall_health = (
                        technical_health * 0.4 +
                        fundamental_health * 0.3 +
                        thesis_health * 0.3
                    )
                    
                    # Update position health
                    position.technical_health = technical_health
                    position.fundamental_health = fundamental_health
                    position.thesis_health = thesis_health
                    position.overall_health = overall_health
                    position.thesis_performance = thesis_status
                    
                    logger.info(f"{symbol} health - Technical: {technical_health:.1f}, Fundamental: {fundamental_health:.1f}, Thesis: {thesis_health:.1f}, Overall: {overall_health:.1f}")
                
                await asyncio.sleep(600)  # Update health every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in health analysis loop: {e}")
                await asyncio.sleep(300)
    
    async def _recommendation_loop(self):
        """Generate recommendations for all positions"""
        while self.running:
            try:
                for symbol, position in self.positions.items():
                    recommendation = await self.health_analyzer.generate_recommendation(position)
                    self.recommendations[symbol] = recommendation
                    
                    logger.info(f"{symbol} recommendation: {recommendation.action} (confidence: {recommendation.confidence}%) - {recommendation.rationale}")
                
                await asyncio.sleep(900)  # Generate recommendations every 15 minutes
                
            except Exception as e:
                logger.error(f"Error in recommendation loop: {e}")
                await asyncio.sleep(300)
    
    async def get_portfolio_status(self) -> Dict:
        """Get current portfolio status for API"""
        return {
            'timestamp': datetime.now().isoformat(),
            'positions': [
                {
                    **asdict(pos),
                    'entry_date': pos.entry_date.isoformat(),
                    'discovery_date': pos.discovery_date.isoformat()
                } for pos in self.positions.values()
            ],
            'recommendations': [
                asdict(rec) for rec in self.recommendations.values()
            ],
            'portfolio_health': asdict(self.portfolio_health) if self.portfolio_health else None
        }

async def main():
    """Main entry point"""
    try:
        portfolio_manager = EnhancedPortfolioManager()
        await portfolio_manager.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
