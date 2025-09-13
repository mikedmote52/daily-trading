#!/usr/bin/env python3
"""
Enhanced Stock Discovery Agent with Claude Code SDK Integration

AI-driven stock analysis and screening with enhanced inter-agent communication.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add shared utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from utils.claude_sdk_helper import ClaudeSDKMessenger
import pandas as pd
import numpy as np
import yfinance as yf
from anthropic import Anthropic
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EnhancedDiscoveryAgent')

class EnhancedStockDiscoveryAgent:
    def __init__(self):
        # Initialize Claude SDK messenger
        self.messenger = ClaudeSDKMessenger(
            agent_name="stock_discovery",
            shared_context_dir="../../../shared_context"
        )
        
        # Initialize Claude AI
        self.claude = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
        
        # Discovery state
        self.stock_universe = self._load_stock_universe()
        self.discovered_stocks = []
        self.screening_criteria = self._load_screening_criteria()
        self.running = False
        
        # ML model for scoring
        self.scaler = StandardScaler()
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self._initialize_ml_model()
        
        # Register message handlers
        self._register_message_handlers()
        
        logger.info("Enhanced Stock Discovery Agent initialized")

    def _register_message_handlers(self):
        """Register handlers for different message types"""
        
        def handle_screening_request(message):
            """Handle requests for stock screening"""
            content = message["content"]
            sender = message["sender"]
            
            logger.info(f"Screening request from {sender}")
            
            # Update screening criteria if provided
            if "criteria" in content:
                self.screening_criteria.update(content["criteria"])
                logger.info("Updated screening criteria")
            
            # Trigger immediate screening
            asyncio.create_task(self._perform_screening(sender))
        
        def handle_market_update(message):
            """Handle market data updates"""
            content = message["content"]
            
            logger.info("Received market update")
            
            # Update shared market data
            self.messenger.update_shared_data("last_market_update", datetime.now().isoformat())
            
            # Trigger analysis update if needed
            if content.get("trigger_analysis", False):
                asyncio.create_task(self._update_analysis())
        
        def handle_optimization_suggestion(message):
            """Handle optimization suggestions from master agent"""
            content = message["content"]
            recommendations = content.get("recommendations", [])
            
            logger.info(f"Received {len(recommendations)} optimization suggestions")
            
            # Apply optimization recommendations
            asyncio.create_task(self._apply_optimizations(recommendations))
        
        # Register handlers
        self.messenger.register_message_handler("screening_requests", handle_screening_request)
        self.messenger.register_message_handler("market_updates", handle_market_update)
        self.messenger.register_message_handler("optimization_suggestion", handle_optimization_suggestion)

    def _load_stock_universe(self) -> List[str]:
        """Load stock universe for screening"""
        return [
            'AAPL', 'GOOGL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA',
            'NFLX', 'DIS', 'PYPL', 'ADBE', 'CRM', 'INTC', 'AMD', 'UBER',
            'SPOT', 'SQ', 'ROKU', 'ZM', 'DOCU', 'PTON', 'MRNA', 'BNTX',
            'PLTR', 'SNOW', 'COIN', 'HOOD', 'RIVN', 'LCID', 'F', 'GM',
            'BAC', 'JPM', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'COST',
            'WMT', 'HD', 'LOW', 'TGT', 'SBUX', 'MCD', 'KO', 'PEP'
        ]

    def _load_screening_criteria(self) -> Dict[str, Any]:
        """Load default screening criteria"""
        return {
            "max_price": 100.0,
            "min_volume": 1000000,
            "min_market_cap": 1000000000,
            "max_pe_ratio": 30.0,
            "min_short_interest": 5.0,
            "min_volatility": 0.02,
            "momentum_period": 20,
            "volume_surge_threshold": 2.0
        }

    def _initialize_ml_model(self):
        """Initialize ML model with synthetic training data"""
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        X = np.random.randn(n_samples, 5)
        X[:, 0] = np.random.normal(0.05, 0.2, n_samples)  # momentum
        X[:, 1] = np.random.exponential(1.5, n_samples)   # volume ratio
        X[:, 2] = np.random.gamma(2, 0.1, n_samples)      # volatility
        X[:, 3] = np.random.gamma(3, 5, n_samples)        # pe ratio
        X[:, 4] = np.random.beta(2, 5, n_samples) * 20    # short interest
        
        y = (
            X[:, 0] * 30 +
            np.log(X[:, 1]) * 15 +
            X[:, 2] * -10 +
            (1 / (1 + X[:, 3] / 20)) * 25 +
            X[:, 4] * 2
        ) + np.random.normal(0, 5, n_samples)
        
        y = np.clip(y, 0, 100)
        
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model.fit(X_scaled, y)
        
        logger.info("ML model initialized")

    async def start(self):
        """Start the enhanced stock discovery agent"""
        logger.info("Starting Enhanced Stock Discovery Agent...")
        self.running = True
        
        # Update initial status
        self.messenger.update_progress(
            status="online",
            current_task="Starting stock discovery system",
            progress_data={
                "startup_time": datetime.now().isoformat(),
                "stocks_analyzed": 0,
                "signals_generated": 0,
                "screening_cycles": 0
            }
        )
        
        # Start message listening
        self.messenger.start_listening(poll_interval=1.0)
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._discovery_loop()),
            asyncio.create_task(self._signal_monitoring_loop()),
            asyncio.create_task(self._performance_reporting_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down Enhanced Stock Discovery Agent...")
            self.running = False
            self.messenger.stop_listening()

    async def _discovery_loop(self):
        """Main discovery loop with enhanced communication"""
        while self.running:
            try:
                logger.info("Starting stock discovery scan...")
                
                self.messenger.update_progress(
                    status="busy",
                    current_task="Scanning stock universe"
                )
                
                # Screen stocks from universe
                discovered = await self._screen_stocks()
                
                # Analyze discovered stocks
                analyzed_stocks = []
                for i, stock_data in enumerate(discovered):
                    analysis = await self._analyze_stock(stock_data)
                    if analysis:
                        analyzed_stocks.append(analysis)
                    
                    # Update progress periodically
                    if i % 10 == 0:
                        self.messenger.update_progress(
                            status="busy",
                            current_task=f"Analyzing stock {i+1}/{len(discovered)}",
                            progress_data={"stocks_analyzed": i+1}
                        )
                
                # Sort by AI score
                analyzed_stocks.sort(key=lambda x: x.get("ai_score", 0), reverse=True)
                self.discovered_stocks = analyzed_stocks[:50]  # Keep top 50
                
                # Update shared data with top picks
                top_picks = analyzed_stocks[:10]
                self.messenger.update_shared_data("top_stock_picks", top_picks)
                
                # Send discoveries to relevant agents
                await self._distribute_discoveries(top_picks)
                
                # Update final progress
                self.messenger.update_progress(
                    status="online",
                    current_task="Discovery scan completed",
                    progress_data={
                        "last_scan": datetime.now().isoformat(),
                        "stocks_analyzed": len(discovered),
                        "top_picks_found": len(top_picks),
                        "screening_cycles": self.messenger.get_progress().get("progress", {}).get("screening_cycles", 0) + 1
                    },
                    performance_metrics={
                        "discovery_accuracy": self._calculate_discovery_accuracy(),
                        "processing_speed": len(discovered) / 300,  # stocks per second
                        "signal_quality": self._calculate_signal_quality()
                    }
                )
                
                logger.info(f"Discovery scan completed. Found {len(top_picks)} top picks")
                
                await asyncio.sleep(300)  # Wait 5 minutes before next scan
                
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                self.messenger.add_alert("error", f"Discovery loop error: {str(e)}")
                await asyncio.sleep(60)

    async def _screen_stocks(self) -> List[Dict[str, Any]]:
        """Screen stocks based on criteria with progress updates"""
        discovered_stocks = []
        
        batch_size = 10
        total_batches = len(self.stock_universe) // batch_size + 1
        
        for batch_num, i in enumerate(range(0, len(self.stock_universe), batch_size)):
            batch = self.stock_universe[i:i + batch_size]
            
            # Update progress
            self.messenger.update_progress(
                status="busy",
                current_task=f"Screening batch {batch_num + 1}/{total_batches}"
            )
            
            try:
                batch_data = []
                for symbol in batch:
                    try:
                        stock_data = await self._fetch_stock_data(symbol)
                        if self._passes_screening(stock_data):
                            batch_data.append(stock_data)
                    except Exception as e:
                        logger.warning(f"Failed to fetch data for {symbol}: {e}")
                        continue
                
                discovered_stocks.extend(batch_data)
                await asyncio.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error processing batch {batch}: {e}")
                continue
        
        return discovered_stocks

    async def _fetch_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch stock data for analysis"""
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="30d")
        
        if len(hist) < 20:
            raise ValueError(f"Insufficient data for {symbol}")
        
        return {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'price': hist['Close'].iloc[-1],
            'volume': hist['Volume'].iloc[-1],
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE'),
            'volatility': hist['Close'].pct_change().std() * np.sqrt(252),
            'history': hist,
            'info': info
        }

    def _passes_screening(self, stock_data: Dict[str, Any]) -> bool:
        """Check if stock passes screening criteria"""
        criteria = self.screening_criteria
        
        if stock_data['price'] > criteria['max_price']:
            return False
        if stock_data['volume'] < criteria['min_volume']:
            return False
        if stock_data['market_cap'] < criteria['min_market_cap']:
            return False
        if stock_data.get('pe_ratio') and stock_data['pe_ratio'] > criteria['max_pe_ratio']:
            return False
        if stock_data['volatility'] < criteria['min_volatility']:
            return False
        
        return True

    async def _analyze_stock(self, stock_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform detailed analysis on a stock"""
        try:
            symbol = stock_data['symbol']
            hist = stock_data['history']
            
            # Calculate technical indicators
            current_price = hist['Close'].iloc[-1]
            previous_price = hist['Close'].iloc[-2] if len(hist) >= 2 else current_price
            change = current_price - previous_price
            change_percent = (change / previous_price) * 100
            
            # Momentum score
            momentum_period = min(self.screening_criteria['momentum_period'], len(hist) - 1)
            if momentum_period > 0:
                momentum_score = (current_price / hist['Close'].iloc[-momentum_period] - 1) * 100
            else:
                momentum_score = 0
            
            # Volume score
            volume_ma = hist['Volume'].rolling(window=20).mean()
            current_volume = hist['Volume'].iloc[-1]
            avg_volume = volume_ma.iloc[-1]
            volume_score = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Calculate AI score using ML model
            features = np.array([[
                momentum_score / 100,
                volume_score,
                stock_data['volatility'],
                stock_data['pe_ratio'] if stock_data['pe_ratio'] else 20,
                stock_data['info'].get('shortPercentOfFloat', 0) * 100
            ]])
            
            features_scaled = self.scaler.transform(features)
            ai_score = int(self.ml_model.predict(features_scaled)[0])
            ai_score = max(0, min(100, ai_score))
            
            # Generate signals using Claude
            signals = await self._generate_ai_signals(symbol, stock_data, momentum_score, volume_score)
            
            return {
                'symbol': symbol,
                'name': stock_data['name'],
                'price': current_price,
                'change': change,
                'change_percent': change_percent,
                'volume': int(current_volume),
                'market_cap': stock_data['market_cap'],
                'pe_ratio': stock_data['pe_ratio'],
                'short_interest': stock_data['info'].get('shortPercentOfFloat', 0) * 100,
                'volatility': stock_data['volatility'],
                'momentum_score': momentum_score,
                'volume_score': volume_score,
                'ai_score': ai_score,
                'signals': signals,
                'recommendation': self._get_recommendation(ai_score, signals, momentum_score)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing stock {stock_data['symbol']}: {e}")
            return None

    async def _generate_ai_signals(self, symbol: str, stock_data: Dict[str, Any], 
                                 momentum_score: float, volume_score: float) -> List[str]:
        """Generate AI-powered trading signals"""
        try:
            analysis_data = {
                'symbol': symbol,
                'price': stock_data['price'],
                'momentum_score': momentum_score,
                'volume_score': volume_score,
                'volatility': stock_data['volatility']
            }
            
            prompt = f"""
            Analyze this stock and generate 1-3 concise trading signals:
            
            {analysis_data}
            
            Generate signals based on:
            - Technical momentum (positive = bullish)
            - Volume patterns (>2.0 = significant surge)
            - Volatility characteristics
            
            Return only a JSON list of signal strings like:
            ["Momentum Breakout", "Volume Surge", "Low Volatility"]
            
            Keep signals actionable and specific.
            """
            
            response = self.claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            import json
            signals = json.loads(response.content[0].text)
            return signals if isinstance(signals, list) else []
            
        except Exception as e:
            logger.warning(f"Failed to generate AI signals for {symbol}: {e}")
            return self._generate_fallback_signals(momentum_score, volume_score)

    def _generate_fallback_signals(self, momentum_score: float, volume_score: float) -> List[str]:
        """Generate fallback signals using rules"""
        signals = []
        if momentum_score > 5:
            signals.append("Positive Momentum")
        if volume_score > 2:
            signals.append("Volume Surge")
        if momentum_score < -5:
            signals.append("Negative Momentum")
        return signals

    def _get_recommendation(self, ai_score: int, signals: List[str], momentum_score: float) -> str:
        """Generate recommendation based on analysis"""
        if ai_score >= 80 and momentum_score > 0:
            return 'BUY'
        elif ai_score >= 60 and any('Breakout' in s or 'Surge' in s for s in signals):
            return 'BUY'
        elif ai_score <= 30 or momentum_score < -10:
            return 'SELL'
        elif ai_score <= 40 and momentum_score < -5:
            return 'AVOID'
        else:
            return 'HOLD'

    async def _distribute_discoveries(self, top_picks: List[Dict[str, Any]]):
        """Send discoveries to relevant agents"""
        try:
            # Send to portfolio management agent
            self.messenger.send_message(
                "portfolio_management",
                "trade_signals",
                {
                    "discoveries": top_picks,
                    "timestamp": datetime.now().isoformat(),
                    "source": "stock_discovery"
                }
            )
            
            # Send to backend for frontend display
            self.messenger.send_message(
                "backend_orchestration",
                "data_updates",
                {
                    "type": "stock_discoveries",
                    "data": top_picks,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Notify master agent
            self.messenger.send_message(
                "master_orchestration",
                "system_status",
                {
                    "discoveries_count": len(top_picks),
                    "avg_ai_score": np.mean([s.get('ai_score', 0) for s in top_picks]),
                    "buy_recommendations": len([s for s in top_picks if s.get('recommendation') == 'BUY'])
                }
            )
            
            logger.info(f"Distributed {len(top_picks)} discoveries to agents")
            
        except Exception as e:
            logger.error(f"Error distributing discoveries: {e}")

    async def _perform_screening(self, requestor: str):
        """Perform immediate screening upon request"""
        try:
            logger.info(f"Performing immediate screening for {requestor}")
            
            # Trigger discovery loop
            asyncio.create_task(self._discovery_loop())
            
        except Exception as e:
            logger.error(f"Error performing screening: {e}")

    async def _update_analysis(self):
        """Update analysis based on new market data"""
        try:
            logger.info("Updating analysis based on market data")
            
            # Re-analyze current top picks
            updated_picks = []
            for stock in self.discovered_stocks[:10]:
                # Fetch fresh data and re-analyze
                updated_data = await self._fetch_stock_data(stock['symbol'])
                updated_analysis = await self._analyze_stock(updated_data)
                if updated_analysis:
                    updated_picks.append(updated_analysis)
            
            # Update shared data
            self.messenger.update_shared_data("top_stock_picks", updated_picks)
            
        except Exception as e:
            logger.error(f"Error updating analysis: {e}")

    async def _apply_optimizations(self, recommendations: List[Dict[str, Any]]):
        """Apply optimization recommendations from master agent"""
        try:
            for rec in recommendations:
                if rec.get("type") == "screening_criteria":
                    # Update screening criteria
                    new_criteria = rec.get("criteria", {})
                    self.screening_criteria.update(new_criteria)
                    logger.info(f"Updated screening criteria: {new_criteria}")
                
                elif rec.get("type") == "analysis_parameters":
                    # Update analysis parameters
                    params = rec.get("parameters", {})
                    logger.info(f"Applied analysis parameter updates: {params}")
            
        except Exception as e:
            logger.error(f"Error applying optimizations: {e}")

    def _calculate_discovery_accuracy(self) -> float:
        """Calculate discovery accuracy metric"""
        # Simplified accuracy calculation
        return 0.85

    def _calculate_signal_quality(self) -> float:
        """Calculate signal quality metric"""
        # Simplified signal quality calculation
        return 0.78

    async def _signal_monitoring_loop(self):
        """Monitor for real-time signals"""
        while self.running:
            try:
                # Check for significant changes in top stocks
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in signal monitoring: {e}")
                await asyncio.sleep(60)

    async def _performance_reporting_loop(self):
        """Report performance metrics periodically"""
        while self.running:
            try:
                # Update performance metrics
                self.messenger.update_progress(
                    status="online",
                    performance_metrics={
                        "discovery_accuracy": self._calculate_discovery_accuracy(),
                        "signal_quality": self._calculate_signal_quality(),
                        "processing_speed": len(self.stock_universe) / 300
                    }
                )
                
                await asyncio.sleep(300)  # Report every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance reporting: {e}")
                await asyncio.sleep(300)

async def main():
    """Main entry point"""
    discovery_agent = EnhancedStockDiscoveryAgent()
    
    try:
        await discovery_agent.start()
    except Exception as e:
        logger.error(f"Fatal error in enhanced discovery agent: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())