#!/usr/bin/env python3
"""
Portfolio Manager with Automatic Stop-Loss
Monitors positions and executes automatic exit strategies
"""
import logging
import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PortfolioManager')

class PortfolioManager:
    """Manages portfolio positions with automatic risk controls"""

    def __init__(self, alpaca_api_key: str = None, alpaca_secret_key: str = None, paper_trading: bool = True):
        self.alpaca_api_key = alpaca_api_key or os.getenv('ALPACA_API_KEY')
        self.alpaca_secret_key = alpaca_secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.paper_trading = paper_trading

        # Risk management parameters
        self.STOP_LOSS_PCT = -15.0  # 15% stop-loss
        self.TRAILING_STOP_PCT = -10.0  # 10% trailing stop for winners
        self.MAX_POSITION_SIZE_PCT = 15.0  # Max 15% of portfolio per position
        self.HIGH_RISK_EXIT_THRESHOLD = -8.0  # Exit "HIGH RISK" positions at -8%

        # Blacklist management
        self.blacklist_file = os.path.join(os.path.dirname(__file__), 'blacklist.json')
        self.blacklist = self.load_blacklist()

        logger.info("✅ Portfolio Manager initialized")
        logger.info(f"🎯 Stop-Loss: {self.STOP_LOSS_PCT}% | Trailing Stop: {self.TRAILING_STOP_PCT}%")

    def load_blacklist(self) -> Dict[str, float]:
        """Load blacklist from file"""
        try:
            if os.path.exists(self.blacklist_file):
                with open(self.blacklist_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Failed to load blacklist: {e}")
            return {}

    def save_blacklist(self):
        """Save blacklist to file"""
        try:
            with open(self.blacklist_file, 'w') as f:
                json.dump(self.blacklist, f, indent=2)
            logger.info(f"💾 Blacklist saved: {len(self.blacklist)} tickers")
        except Exception as e:
            logger.error(f"Failed to save blacklist: {e}")

    def add_to_blacklist(self, symbol: str, days: int = 30):
        """Add ticker to blacklist with expiration"""
        expiry_timestamp = datetime.now().timestamp() + (days * 24 * 3600)
        self.blacklist[symbol] = expiry_timestamp
        self.save_blacklist()
        logger.info(f"🚫 Added {symbol} to blacklist for {days} days")

    def is_blacklisted(self, symbol: str) -> bool:
        """Check if symbol is currently blacklisted"""
        if symbol not in self.blacklist:
            return False

        current_time = datetime.now().timestamp()
        if current_time > self.blacklist[symbol]:
            # Expired, remove from blacklist
            del self.blacklist[symbol]
            self.save_blacklist()
            return False

        return True

    def check_position_risk(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate risk level of a position"""
        symbol = position.get('symbol')
        unrealized_pnl_pct = position.get('unrealized_pnl_pct', 0)
        current_price = position.get('current_price', 0)
        avg_entry_price = position.get('avg_price', 0)

        # Calculate risk metrics
        risk_assessment = {
            'symbol': symbol,
            'action': 'HOLD',
            'reason': 'Within normal parameters',
            'risk_level': 'NORMAL',
            'should_exit': False
        }

        # Stop-loss check
        if unrealized_pnl_pct <= self.STOP_LOSS_PCT:
            risk_assessment.update({
                'action': 'SELL',
                'reason': f'Stop-loss triggered at {unrealized_pnl_pct:.1f}%',
                'risk_level': 'CRITICAL',
                'should_exit': True
            })
            return risk_assessment

        # High risk threshold for deteriorating positions
        if unrealized_pnl_pct <= self.HIGH_RISK_EXIT_THRESHOLD:
            risk_assessment.update({
                'action': 'SELL',
                'reason': f'High risk exit at {unrealized_pnl_pct:.1f}%',
                'risk_level': 'HIGH',
                'should_exit': True
            })
            return risk_assessment

        # Monitor closely if losing
        if unrealized_pnl_pct < -5.0:
            risk_assessment.update({
                'action': 'MONITOR',
                'reason': f'Position down {unrealized_pnl_pct:.1f}% - monitor closely',
                'risk_level': 'MODERATE'
            })

        # Trailing stop for winners
        if unrealized_pnl_pct > 20.0:
            # Check if price has fallen from peak
            peak_price = avg_entry_price * (1 + unrealized_pnl_pct / 100)
            price_drop_from_peak = ((current_price - peak_price) / peak_price) * 100

            if price_drop_from_peak <= self.TRAILING_STOP_PCT:
                risk_assessment.update({
                    'action': 'SELL',
                    'reason': f'Trailing stop: {price_drop_from_peak:.1f}% from peak',
                    'risk_level': 'PROFIT_PROTECTION',
                    'should_exit': True
                })

        return risk_assessment

    def analyze_portfolio(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze entire portfolio and generate action recommendations"""
        logger.info(f"📊 Analyzing portfolio: {len(positions)} positions")

        actions = []
        positions_to_exit = []
        positions_to_monitor = []
        high_risk_count = 0

        for position in positions:
            symbol = position.get('symbol')
            risk_assessment = self.check_position_risk(position)

            if risk_assessment['should_exit']:
                positions_to_exit.append(risk_assessment)
                logger.warning(f"🚨 {symbol}: {risk_assessment['reason']}")
            elif risk_assessment['risk_level'] == 'MODERATE':
                positions_to_monitor.append(risk_assessment)
                logger.info(f"⚠️  {symbol}: {risk_assessment['reason']}")
            elif risk_assessment['risk_level'] == 'HIGH':
                high_risk_count += 1

            actions.append(risk_assessment)

        # Generate summary report
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_positions': len(positions),
            'positions_to_exit': len(positions_to_exit),
            'positions_to_monitor': len(positions_to_monitor),
            'high_risk_positions': high_risk_count,
            'actions': actions,
            'immediate_exits': positions_to_exit,
            'monitor_list': positions_to_monitor
        }

        logger.info(f"✅ Analysis complete: {len(positions_to_exit)} exits, {len(positions_to_monitor)} monitoring")

        return summary

    def execute_exit_strategy(self, position: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Execute exit for a position (integrates with Alpaca)"""
        symbol = position.get('symbol')
        shares = position.get('shares', 0)

        logger.info(f"📤 Executing exit for {symbol}: {shares} shares")
        logger.info(f"   Reason: {reason}")

        # Add to blacklist after exit
        self.add_to_blacklist(symbol, days=30)

        # TODO: Integrate with actual Alpaca API
        # For now, return mock execution result
        result = {
            'symbol': symbol,
            'action': 'SELL',
            'shares': shares,
            'status': 'SIMULATED',  # Will be 'EXECUTED' when Alpaca integrated
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'blacklisted': True
        }

        logger.info(f"✅ Exit executed for {symbol}")
        return result

    def get_blacklist_status(self) -> Dict[str, Any]:
        """Get current blacklist with expiration times"""
        current_time = datetime.now().timestamp()

        active_blacklist = {}
        for symbol, expiry in self.blacklist.items():
            if expiry > current_time:
                days_remaining = (expiry - current_time) / (24 * 3600)
                active_blacklist[symbol] = {
                    'expires_at': datetime.fromtimestamp(expiry).isoformat(),
                    'days_remaining': round(days_remaining, 1)
                }

        return {
            'blacklisted_tickers': active_blacklist,
            'total_blacklisted': len(active_blacklist)
        }

def main():
    """Test portfolio manager with sample data"""
    manager = PortfolioManager()

    # Sample positions from the portfolio analysis
    sample_positions = [
        {'symbol': 'GCCS', 'shares': 10, 'avg_price': 9.50, 'current_price': 9.35, 'unrealized_pnl_pct': -1.58},
        {'symbol': 'CDLX', 'shares': 35, 'avg_price': 2.95, 'current_price': 2.60, 'unrealized_pnl_pct': -12.9},
        {'symbol': 'CLOV', 'shares': 31, 'avg_price': 3.18, 'current_price': 3.12, 'unrealized_pnl_pct': -2.0},
        {'symbol': 'FATN', 'shares': 20, 'avg_price': 8.44, 'current_price': 6.38, 'unrealized_pnl_pct': -34.4},
        {'symbol': 'IQ', 'shares': 37, 'avg_price': 2.59, 'current_price': 2.61, 'unrealized_pnl_pct': 4.8},
        {'symbol': 'LAES', 'shares': 23, 'avg_price': 4.24, 'current_price': 3.73, 'unrealized_pnl_pct': -12.0},
        {'symbol': 'LASE', 'shares': 22, 'avg_price': 4.48, 'current_price': 4.15, 'unrealized_pnl_pct': 7.4},
        {'symbol': 'QMCO', 'shares': 8, 'avg_price': 11.93, 'current_price': 9.46, 'unrealized_pnl_pct': -20.7},
        {'symbol': 'QSI', 'shares': 68, 'avg_price': 1.49, 'current_price': 1.38, 'unrealized_pnl_pct': -27.2},
    ]

    # Analyze portfolio
    analysis = manager.analyze_portfolio(sample_positions)

    # Print summary
    print("\n" + "="*60)
    print("📊 PORTFOLIO RISK ANALYSIS")
    print("="*60)
    print(f"Total Positions: {analysis['total_positions']}")
    print(f"Immediate Exits Required: {analysis['positions_to_exit']}")
    print(f"Positions to Monitor: {analysis['positions_to_monitor']}")
    print(f"High Risk Positions: {analysis['high_risk_positions']}")

    # Show immediate exits
    if analysis['immediate_exits']:
        print("\n🚨 IMMEDIATE EXITS:")
        for exit_action in analysis['immediate_exits']:
            print(f"   {exit_action['symbol']}: {exit_action['reason']}")

    # Show blacklist status
    print("\n🚫 BLACKLIST STATUS:")
    blacklist_status = manager.get_blacklist_status()
    for symbol, info in blacklist_status['blacklisted_tickers'].items():
        print(f"   {symbol}: {info['days_remaining']} days remaining")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()