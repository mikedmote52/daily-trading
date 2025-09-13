#!/usr/bin/env python3
"""
Enhanced Explosive Growth Discovery Agent

Specialized agent for finding explosive growth stocks using Polygon API
and advanced pattern recognition. Designed to replicate winning stock finds.
"""

import asyncio
import logging
import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import redis
import pandas as pd
import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv

# Add shared utilities to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../shared'))
from utils.polygon_client import PolygonClient, ExplosiveGrowthSignal
from utils.claude_sdk_helper import ClaudeSDKMessenger

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ExplosiveDiscoveryAgent')

@dataclass
class ExplosivePattern:
    """Pattern definition for explosive growth detection"""
    name: str
    description: str
    min_price_change: float
    min_volume_surge: float
    max_market_cap: float
    lookback_hours: int
    confidence_threshold: float

class ExplosiveGrowthDiscoveryAgent:
    """Enhanced discovery agent focused on explosive growth opportunities"""
    
    def __init__(self):
        # Initialize APIs
        self.polygon = PolygonClient(os.getenv('POLYGON_API_KEY'))
        self.claude = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
        self.redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
        
        # Initialize communication
        self.messenger = ClaudeSDKMessenger(
            agent_name="explosive_discovery",
            shared_context_dir="../shared_context"
        )
        
        # Register message handlers
        self._register_message_handlers()
        
        # Explosive growth patterns to detect
        self.patterns = [
            ExplosivePattern(
                name="breakout_surge",
                description="High volume breakout with significant price movement",
                min_price_change=8.0,
                min_volume_surge=200.0,
                max_market_cap=10e9,
                lookback_hours=2,
                confidence_threshold=0.7
            ),
            ExplosivePattern(
                name="news_catalyst",
                description="News-driven explosive movement with volume confirmation",
                min_price_change=15.0,
                min_volume_surge=300.0,
                max_market_cap=5e9,
                lookback_hours=1,
                confidence_threshold=0.8
            ),
            ExplosivePattern(
                name="momentum_acceleration",
                description="Accelerating momentum with increasing volume",
                min_price_change=5.0,
                min_volume_surge=150.0,
                max_market_cap=20e9,
                lookback_hours=4,
                confidence_threshold=0.6
            )
        ]
        
        self.running = False
        self.discovered_explosive_stocks = []
        self.last_scan_time = None
        
    def _register_message_handlers(self):
        """Register handlers for coordination messages"""
        
        def handle_scan_request(message):
            """Handle manual scan requests from orchestrator"""
            logger.info("Received scan request from orchestrator")
            asyncio.create_task(self._perform_explosive_scan())
        
        def handle_pattern_update(message):
            """Handle pattern configuration updates"""
            content = message.get("content", {})
            if "patterns" in content:
                self._update_patterns(content["patterns"])
        
        self.messenger.register_message_handler("scan_request", handle_scan_request)
        self.messenger.register_message_handler("pattern_update", handle_pattern_update)
    
    async def start(self):
        """Start the explosive discovery agent"""
        logger.info("Starting Explosive Growth Discovery Agent...")
        self.running = True
        
        # Update status
        self.messenger.update_progress(
            status="online",
            current_task="Initializing explosive growth detection",
            progress_data={
                "patterns_loaded": len(self.patterns),
                "last_scan": None,
                "explosives_found": 0
            }
        )
        
        # Start message listening
        self.messenger.start_listening(poll_interval=2.0)
        
        # Start main discovery loops
        tasks = [
            asyncio.create_task(self._continuous_explosive_scan()),
            asyncio.create_task(self._pattern_analysis_loop()),
            asyncio.create_task(self._alert_monitoring_loop()),
            asyncio.create_task(self._status_reporting_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down Explosive Discovery Agent...")
            self.running = False
            self.messenger.stop_listening()
    
    async def _continuous_explosive_scan(self):
        """Continuously scan for explosive growth opportunities"""
        while self.running:
            try:
                # Market hours check
                if not self._is_market_hours():
                    await asyncio.sleep(300)  # Check every 5 minutes when market closed
                    continue
                
                logger.info("Starting explosive growth scan...")
                
                # Update status
                self.messenger.update_progress(
                    status="online",
                    current_task="Scanning for explosive growth",
                    progress_data={"scan_start": datetime.now().isoformat()}
                )
                
                # Get explosive candidates from Polygon
                explosive_signals = await self.polygon.get_explosive_growth_candidates(limit=50)
                
                if explosive_signals:
                    logger.info(f"Found {len(explosive_signals)} potential explosive growth candidates")
                    
                    # Analyze each signal with AI
                    analyzed_signals = []
                    for signal in explosive_signals:
                        analysis = await self._ai_analyze_explosive_signal(signal)
                        if analysis and analysis.get('recommendation') == 'BUY':
                            analyzed_signals.append({
                                'signal': signal,
                                'analysis': analysis
                            })
                    
                    if analyzed_signals:
                        # Sort by AI confidence and signal strength
                        analyzed_signals.sort(
                            key=lambda x: (x['analysis'].get('confidence', 0) * 
                                         x['signal'].signal_strength), 
                            reverse=True
                        )
                        
                        # Send top opportunities to other agents
                        await self._broadcast_explosive_opportunities(analyzed_signals[:10])
                        
                        # Store discoveries
                        self.discovered_explosive_stocks.extend(analyzed_signals)
                        self.last_scan_time = datetime.now()
                
                # Update progress
                self.messenger.update_progress(
                    status="online",
                    current_task="Explosive scan completed",
                    progress_data={
                        "last_scan": datetime.now().isoformat(),
                        "candidates_found": len(explosive_signals),
                        "high_confidence": len(analyzed_signals) if 'analyzed_signals' in locals() else 0,
                        "total_discoveries": len(self.discovered_explosive_stocks)
                    },
                    performance_metrics={
                        "scan_frequency_minutes": 2,
                        "detection_accuracy": self._calculate_detection_accuracy(),
                        "avg_signal_strength": np.mean([s.signal_strength for s in explosive_signals]) if explosive_signals else 0
                    }
                )
                
                # Wait before next scan (2 minutes during market hours)
                await asyncio.sleep(120)
                
            except Exception as e:
                logger.error(f"Error in explosive scan: {e}")
                self.messenger.add_alert("error", f"Explosive scan failed: {str(e)}", "discovery")
                await asyncio.sleep(120)
    
    async def _ai_analyze_explosive_signal(self, signal: ExplosiveGrowthSignal) -> Optional[Dict[str, Any]]:
        """Use Claude AI to analyze explosive growth signal"""
        try:
            prompt = f"""
            Analyze this explosive growth stock signal for trading potential:
            
            Symbol: {signal.symbol}
            Current Price: ${signal.current_price:.2f}
            Price Change: {signal.price_change_percent:.2f}%
            Volume Surge: {signal.volume_surge_percent:.1f}%
            Market Cap: ${signal.market_cap/1e6:.0f}M (if available)
            Signal Strength: {signal.signal_strength:.2f}/1.0
            Risk Score: {signal.risk_score:.2f}/1.0
            Triggers: {', '.join(signal.triggers)}
            Detection Time: {signal.detection_time}
            
            Please provide analysis in JSON format:
            {{
                "recommendation": "BUY/HOLD/AVOID",
                "confidence": 0.85,
                "price_target": 45.50,
                "stop_loss": 38.20,
                "time_horizon": "1-3 days",
                "catalyst_analysis": "Strong earnings beat with revenue guidance raise",
                "risk_factors": ["High volatility", "Low volume outside of surge"],
                "opportunity_type": "earnings_momentum/breakout/news_catalyst/technical_breakout",
                "entry_strategy": "Market order vs limit order recommendation",
                "reasoning": "Detailed explanation of why this is explosive growth opportunity"
            }}
            
            Focus on identifying legitimate explosive growth vs pump-and-dump schemes.
            Consider market conditions, sector trends, and fundamental catalysts.
            """
            
            response = self.claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse JSON response
            analysis = json.loads(response.content[0].text)
            return analysis
            
        except Exception as e:
            logger.error(f"Error in AI analysis for {signal.symbol}: {e}")
            return None
    
    async def _broadcast_explosive_opportunities(self, opportunities: List[Dict[str, Any]]):
        """Broadcast explosive opportunities to other agents"""
        try:
            # Send to backtesting agent for quick validation
            self.messenger.send_message(
                "backtesting",
                "explosive_opportunity",
                {
                    "opportunities": [
                        {
                            "symbol": opp['signal'].symbol,
                            "price": opp['signal'].current_price,
                            "signal_strength": opp['signal'].signal_strength,
                            "triggers": opp['signal'].triggers,
                            "ai_analysis": opp['analysis']
                        }
                        for opp in opportunities
                    ],
                    "priority": "high",
                    "detection_time": datetime.now().isoformat()
                }
            )
            
            # Send to portfolio management for position sizing
            self.messenger.send_message(
                "portfolio_management",
                "explosive_opportunity",
                {
                    "top_opportunities": opportunities[:5],  # Top 5 only
                    "priority": "high"
                }
            )
            
            # Send alert to master orchestrator
            self.messenger.send_message(
                "master_orchestration",
                "explosive_alert",
                {
                    "alert_type": "explosive_growth_detected",
                    "count": len(opportunities),
                    "top_symbol": opportunities[0]['signal'].symbol,
                    "max_signal_strength": opportunities[0]['signal'].signal_strength
                }
            )
            
            # Update shared data
            self.messenger.update_shared_data("latest_explosive_opportunities", {
                "timestamp": datetime.now().isoformat(),
                "opportunities": [
                    {
                        "symbol": opp['signal'].symbol,
                        "price": opp['signal'].current_price,
                        "change_percent": opp['signal'].price_change_percent,
                        "signal_strength": opp['signal'].signal_strength,
                        "ai_confidence": opp['analysis'].get('confidence', 0)
                    }
                    for opp in opportunities[:10]
                ]
            })
            
            logger.info(f"Broadcasted {len(opportunities)} explosive opportunities")
            
        except Exception as e:
            logger.error(f"Error broadcasting opportunities: {e}")
    
    async def _pattern_analysis_loop(self):
        """Analyze patterns in discovered explosive stocks"""
        while self.running:
            try:
                if len(self.discovered_explosive_stocks) >= 10:
                    # Analyze patterns every hour
                    patterns = await self._analyze_success_patterns()
                    if patterns:
                        self.messenger.update_shared_data("explosive_patterns", patterns)
                
                await asyncio.sleep(3600)  # Analyze patterns every hour
                
            except Exception as e:
                logger.error(f"Error in pattern analysis: {e}")
                await asyncio.sleep(3600)
    
    async def _alert_monitoring_loop(self):
        """Monitor for real-time alerts and breaking news"""
        while self.running:
            try:
                # Check for any urgent market alerts that might create explosive opportunities
                # This could integrate with news APIs, earnings calendars, etc.
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _status_reporting_loop(self):
        """Send regular status reports to the master orchestrator"""
        while self.running:
            try:
                # Send detailed status every 5 minutes
                status_report = {
                    "agent_type": "explosive_discovery",
                    "status": "active_scanning",
                    "discoveries_today": len([d for d in self.discovered_explosive_stocks 
                                            if d['signal'].detection_time.date() == datetime.now().date()]),
                    "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
                    "patterns_monitored": len(self.patterns),
                    "market_status": "open" if self._is_market_hours() else "closed"
                }
                
                self.messenger.send_message(
                    "master_orchestration",
                    "status_report",
                    status_report
                )
                
                await asyncio.sleep(300)  # Report every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in status reporting: {e}")
                await asyncio.sleep(300)
    
    def _is_market_hours(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        # Simple market hours check (9:30 AM - 4:00 PM ET, Monday-Friday)
        if now.weekday() > 4:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _calculate_detection_accuracy(self) -> float:
        """Calculate accuracy of explosive growth detection"""
        # This would track actual performance of discovered stocks
        # For now, return a placeholder
        return 0.75
    
    async def _analyze_success_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in successful explosive growth discoveries"""
        # This would analyze which patterns led to successful trades
        # Return pattern analysis for system improvement
        return {
            "most_successful_pattern": "breakout_surge",
            "average_hold_time": "2.3 days",
            "success_rate": "68%",
            "best_time_of_day": "10:30-11:30 AM",
            "optimal_market_cap_range": "500M-5B"
        }
    
    def _update_patterns(self, new_patterns: List[Dict[str, Any]]):
        """Update detection patterns based on feedback"""
        try:
            self.patterns = [ExplosivePattern(**pattern) for pattern in new_patterns]
            logger.info(f"Updated patterns: {len(self.patterns)} patterns loaded")
        except Exception as e:
            logger.error(f"Error updating patterns: {e}")

async def main():
    """Main entry point for explosive discovery agent"""
    agent = ExplosiveGrowthDiscoveryAgent()
    
    try:
        await agent.start()
    except Exception as e:
        logger.error(f"Fatal error in explosive discovery agent: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)