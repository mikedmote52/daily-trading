#!/usr/bin/env python3
"""
Communication Monitor

Real-time monitoring of inter-agent communication and system health.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
import time
import logging

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils.claude_sdk_helper import ClaudeSDKMessenger
from shared.communication.message_router import MessageRouter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CommMonitor')

class CommunicationMonitor:
    """Monitor and display real-time communication between agents"""
    
    def __init__(self):
        self.messenger = ClaudeSDKMessenger("communication_monitor")
        self.router = MessageRouter()
        self.running = False
        
        # Display settings
        self.max_lines = 50
        self.refresh_interval = 2.0
        
    async def start_monitoring(self):
        """Start real-time communication monitoring"""
        print("🔍 Starting Communication Monitor...")
        print("=" * 80)
        
        self.running = True
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._display_loop()),
            asyncio.create_task(self._statistics_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\n👋 Communication Monitor stopped")
            self.running = False

    async def _display_loop(self):
        """Display real-time communication data"""
        while self.running:
            try:
                # Clear screen
                print("\033[2J\033[H", end="")
                
                # Header
                self._print_header()
                
                # System status
                await self._print_system_status()
                
                # Recent messages
                await self._print_recent_messages()
                
                # Communication statistics
                await self._print_communication_stats()
                
                await asyncio.sleep(self.refresh_interval)
                
            except Exception as e:
                logger.error(f"Error in display loop: {e}")
                await asyncio.sleep(self.refresh_interval)

    def _print_header(self):
        """Print monitoring header"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"🔍 Communication Monitor - {timestamp}")
        print("=" * 80)

    async def _print_system_status(self):
        """Print overall system status"""
        try:
            health = self.messenger.get_system_health()
            progress = self.messenger.get_progress()
            
            print(f"📊 SYSTEM STATUS")
            print(f"   Health: {self._colorize_health(health['health'])}")
            print(f"   Active Agents: {health['active_agents']}/{health['total_agents']}")
            print(f"   Last Update: {health.get('last_update', 'Never')}")
            print(f"   Alerts: {health.get('alerts_count', 0)}")
            print()
            
        except Exception as e:
            print(f"   Error getting system status: {e}")
            print()

    async def _print_recent_messages(self):
        """Print recent inter-agent messages"""
        try:
            # Get recent communication log
            progress = self.messenger.get_progress()
            comm_log = progress.get("communication_log", [])
            
            print(f"📨 RECENT MESSAGES (last {min(10, len(comm_log))})")
            print("-" * 40)
            
            if comm_log:
                for entry in comm_log[-10:]:
                    timestamp = entry.get("timestamp", "")[:19]  # Remove microseconds
                    from_agent = entry.get("from", "unknown")
                    to_agent = entry.get("to", "unknown")
                    msg_type = entry.get("type", "unknown")
                    
                    print(f"   {timestamp} | {from_agent:15} → {to_agent:15} | {msg_type}")
            else:
                print("   No recent messages")
            
            print()
            
        except Exception as e:
            print(f"   Error getting recent messages: {e}")
            print()

    async def _print_communication_stats(self):
        """Print communication statistics"""
        try:
            # Get router statistics
            stats = self.router.get_routing_statistics()
            analysis = self.router.analyze_communication_patterns()
            
            print(f"📈 COMMUNICATION STATISTICS")
            print("-" * 40)
            print(f"   Total Messages: {stats['statistics']['total_messages']}")
            print(f"   Messages Routed: {stats['statistics']['messages_routed']}")
            print(f"   Failed Routes: {stats['statistics']['failed_routes']}")
            print(f"   Active Agents: {stats['active_agents']}")
            
            if analysis.get("status") == "success":
                patterns = analysis["patterns"]
                
                # Most active sender
                most_active = patterns.get("most_active_sender", {})
                if most_active.get("count", 0) > 0:
                    print(f"   Most Active: {most_active['agent']} ({most_active['count']} msgs)")
                
                # Communication frequency
                frequency = patterns.get("communication_frequency", {})
                if frequency.get("messages_per_hour", 0) > 0:
                    print(f"   Frequency: {frequency['messages_per_hour']:.1f} msgs/hour")
                
                # Insights
                insights = analysis.get("insights", [])
                if insights:
                    print(f"   💡 Insights:")
                    for insight in insights[:2]:  # Show top 2 insights
                        print(f"      • {insight}")
            
            print()
            
        except Exception as e:
            print(f"   Error getting communication stats: {e}")
            print()

    def _colorize_health(self, health: str) -> str:
        """Add color coding to health status"""
        colors = {
            "healthy": "\033[92m",    # Green
            "degraded": "\033[93m",   # Yellow
            "critical": "\033[91m",   # Red
            "unknown": "\033[90m"     # Gray
        }
        reset = "\033[0m"
        
        color = colors.get(health, "")
        return f"{color}{health.upper()}{reset}"

    async def _statistics_loop(self):
        """Background task to update statistics"""
        while self.running:
            try:
                # Perform any background statistics collection
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in statistics loop: {e}")
                await asyncio.sleep(10)

    async def show_detailed_analysis(self):
        """Show detailed communication analysis"""
        print("\n🔬 DETAILED COMMUNICATION ANALYSIS")
        print("=" * 60)
        
        try:
            analysis = self.router.analyze_communication_patterns()
            
            if analysis.get("status") == "success":
                patterns = analysis["patterns"]
                insights = analysis["insights"]
                
                # Message type distribution
                msg_types = patterns.get("message_type_distribution", {})
                if msg_types:
                    print("\n📊 Message Type Distribution:")
                    for msg_type, count in sorted(msg_types.items(), key=lambda x: x[1], reverse=True):
                        print(f"   {msg_type:20} : {count:4} messages")
                
                # Communication frequency analysis
                frequency = patterns.get("communication_frequency", {})
                if frequency:
                    print(f"\n⏱️  Communication Frequency:")
                    print(f"   Messages per hour: {frequency.get('messages_per_hour', 0):.2f}")
                    print(f"   Peak hour: {frequency.get('peak_hour', 0):02d}:00")
                
                # Response time analysis
                response_times = patterns.get("response_times", {})
                if response_times:
                    print(f"\n⚡ Response Times:")
                    print(f"   Average: {response_times.get('avg_response_time_seconds', 0):.2f}s")
                    print(f"   Median: {response_times.get('median_response_time_seconds', 0):.2f}s")
                
                # Insights
                if insights:
                    print(f"\n💡 Communication Insights:")
                    for i, insight in enumerate(insights, 1):
                        print(f"   {i}. {insight}")
                
                # Optimization recommendations
                optimizations = self.router.optimize_routing()
                if optimizations:
                    print(f"\n🔧 Optimization Recommendations:")
                    for i, opt in enumerate(optimizations, 1):
                        print(f"   {i}. {opt}")
            
            else:
                print(f"   Analysis unavailable: {analysis.get('error', 'No data')}")
        
        except Exception as e:
            print(f"   Error generating detailed analysis: {e}")

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor inter-agent communication")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed analysis instead of real-time monitoring"
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=2.0,
        help="Refresh interval in seconds (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    monitor = CommunicationMonitor()
    monitor.refresh_interval = args.refresh
    
    try:
        if args.detailed:
            await monitor.show_detailed_analysis()
        else:
            await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())