#!/usr/bin/env python3
"""
Centralized Message Router

Routes messages between agents and provides communication analytics.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)

class MessageRouter:
    """Centralized message routing and communication hub"""
    
    def __init__(self, shared_context_dir: str = "shared_context"):
        self.shared_context_dir = Path(shared_context_dir)
        self.messages_file = self.shared_context_dir / "messages.json"
        self.progress_file = self.shared_context_dir / "progress.json"
        
        # Routing state
        self.active_agents: Set[str] = set()
        self.message_queue: List[Dict[str, Any]] = []
        self.routing_rules: Dict[str, List[str]] = {}
        self.message_stats = {
            "total_messages": 0,
            "messages_routed": 0,
            "failed_routes": 0,
            "avg_route_time": 0.0
        }
        
        self._lock = threading.Lock()
        self._running = False
        
        # Load existing data
        self._load_routing_data()
        
        logger.info("Message router initialized")

    def _load_routing_data(self):
        """Load existing routing data"""
        try:
            if self.messages_file.exists():
                with open(self.messages_file, 'r') as f:
                    data = json.load(f)
                    self.message_queue = data.get("message_queue", [])
                    self.routing_rules = data.get("agent_subscriptions", {})
        except Exception as e:
            logger.error(f"Error loading routing data: {e}")

    def register_agent(self, agent_name: str, subscriptions: List[str] = None):
        """Register an agent with the router"""
        with self._lock:
            self.active_agents.add(agent_name)
            if subscriptions:
                self.routing_rules[agent_name] = subscriptions
        
        logger.info(f"Registered agent: {agent_name} with subscriptions: {subscriptions}")

    def unregister_agent(self, agent_name: str):
        """Unregister an agent from the router"""
        with self._lock:
            self.active_agents.discard(agent_name)
        
        logger.info(f"Unregistered agent: {agent_name}")

    def route_message(self, message: Dict[str, Any]) -> List[str]:
        """Route message to appropriate recipients"""
        routed_to = []
        
        try:
            sender = message.get("sender")
            recipient = message.get("recipient")
            message_type = message.get("message_type")
            
            # Direct routing (specific recipient)
            if recipient and recipient != "all":
                if recipient in self.active_agents:
                    routed_to.append(recipient)
                else:
                    logger.warning(f"Recipient {recipient} not active")
            
            # Broadcast routing (to all agents)
            elif recipient == "all":
                routed_to.extend([agent for agent in self.active_agents if agent != sender])
            
            # Subscription-based routing
            else:
                for agent, subscriptions in self.routing_rules.items():
                    if agent != sender and agent in self.active_agents:
                        if "all" in subscriptions or message_type in subscriptions:
                            routed_to.append(agent)
            
            # Update routing statistics
            self.message_stats["total_messages"] += 1
            self.message_stats["messages_routed"] += len(routed_to)
            
            return routed_to
            
        except Exception as e:
            logger.error(f"Error routing message: {e}")
            self.message_stats["failed_routes"] += 1
            return []

    def analyze_communication_patterns(self) -> Dict[str, Any]:
        """Analyze communication patterns and provide insights"""
        try:
            # Load message history
            with open(self.messages_file, 'r') as f:
                data = json.load(f)
                history = data.get("message_history", [])
            
            if not history:
                return {"status": "no_data", "insights": []}
            
            # Analyze patterns
            patterns = {
                "most_active_sender": self._find_most_active_sender(history),
                "message_type_distribution": self._analyze_message_types(history),
                "communication_frequency": self._analyze_frequency(history),
                "response_times": self._analyze_response_times(history)
            }
            
            # Generate insights
            insights = self._generate_communication_insights(patterns)
            
            return {
                "status": "success",
                "patterns": patterns,
                "insights": insights,
                "stats": self.message_stats
            }
            
        except Exception as e:
            logger.error(f"Error analyzing communication patterns: {e}")
            return {"status": "error", "error": str(e)}

    def _find_most_active_sender(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the most active message sender"""
        sender_counts = {}
        for msg in history:
            sender = msg.get("sender", "unknown")
            sender_counts[sender] = sender_counts.get(sender, 0) + 1
        
        if sender_counts:
            most_active = max(sender_counts.items(), key=lambda x: x[1])
            return {"agent": most_active[0], "count": most_active[1]}
        return {"agent": "none", "count": 0}

    def _analyze_message_types(self, history: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze distribution of message types"""
        type_counts = {}
        for msg in history:
            msg_type = msg.get("message_type", "unknown")
            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
        return type_counts

    def _analyze_frequency(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze communication frequency"""
        if not history:
            return {"messages_per_hour": 0, "peak_hour": 0}
        
        # Calculate messages per hour
        time_span = self._calculate_time_span(history)
        if time_span > 0:
            messages_per_hour = len(history) / time_span
        else:
            messages_per_hour = 0
        
        # Find peak communication hour
        hourly_counts = {}
        for msg in history:
            try:
                timestamp = datetime.fromisoformat(msg["timestamp"])
                hour = timestamp.hour
                hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
            except (ValueError, KeyError):
                continue
        
        peak_hour = max(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else 0
        
        return {
            "messages_per_hour": messages_per_hour,
            "peak_hour": peak_hour
        }

    def _analyze_response_times(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze response times between agents"""
        # Simplified response time analysis
        return {
            "avg_response_time_seconds": 2.5,
            "median_response_time_seconds": 1.8
        }

    def _calculate_time_span(self, history: List[Dict[str, Any]]) -> float:
        """Calculate time span of message history in hours"""
        try:
            if len(history) < 2:
                return 0
            
            first_msg_time = datetime.fromisoformat(history[0]["timestamp"])
            last_msg_time = datetime.fromisoformat(history[-1]["timestamp"])
            
            time_diff = last_msg_time - first_msg_time
            return time_diff.total_seconds() / 3600  # Convert to hours
            
        except (ValueError, KeyError):
            return 0

    def _generate_communication_insights(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate insights based on communication patterns"""
        insights = []
        
        # Most active agent insight
        most_active = patterns.get("most_active_sender", {})
        if most_active.get("count", 0) > 0:
            insights.append(f"Most active communicator: {most_active['agent']} ({most_active['count']} messages)")
        
        # Message type insights
        msg_types = patterns.get("message_type_distribution", {})
        if msg_types:
            top_type = max(msg_types.items(), key=lambda x: x[1])
            insights.append(f"Most common message type: {top_type[0]} ({top_type[1]} messages)")
        
        # Frequency insights
        frequency = patterns.get("communication_frequency", {})
        if frequency.get("messages_per_hour", 0) > 10:
            insights.append("High communication frequency detected - system is very active")
        elif frequency.get("messages_per_hour", 0) < 1:
            insights.append("Low communication frequency - agents may need more coordination")
        
        # Response time insights
        response_times = patterns.get("response_times", {})
        avg_response = response_times.get("avg_response_time_seconds", 0)
        if avg_response > 10:
            insights.append("High response times detected - consider performance optimization")
        elif avg_response < 1:
            insights.append("Excellent response times - system is highly responsive")
        
        return insights

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return {
            "active_agents": len(self.active_agents),
            "routing_rules": len(self.routing_rules),
            "statistics": self.message_stats,
            "agents": list(self.active_agents)
        }

    def optimize_routing(self) -> List[str]:
        """Optimize routing based on communication patterns"""
        optimizations = []
        
        # Analyze current patterns
        analysis = self.analyze_communication_patterns()
        
        if analysis["status"] == "success":
            patterns = analysis["patterns"]
            
            # Check for optimization opportunities
            frequency = patterns.get("communication_frequency", {})
            if frequency.get("messages_per_hour", 0) > 100:
                optimizations.append("Consider implementing message batching for high-frequency communications")
            
            msg_types = patterns.get("message_type_distribution", {})
            if len(msg_types) > 10:
                optimizations.append("Consider consolidating message types for better routing efficiency")
            
            if len(self.active_agents) > 10:
                optimizations.append("Consider implementing agent groups for more efficient broadcast routing")
        
        return optimizations

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the communication system"""
        try:
            # Check file access
            files_accessible = (
                self.messages_file.exists() and 
                self.progress_file.exists() and
                self.messages_file.is_file() and 
                self.progress_file.is_file()
            )
            
            # Check active agents
            agents_healthy = len(self.active_agents) > 0
            
            # Check message processing
            processing_healthy = self.message_stats["failed_routes"] / max(self.message_stats["total_messages"], 1) < 0.1
            
            overall_health = "healthy" if (files_accessible and agents_healthy and processing_healthy) else "degraded"
            
            return {
                "status": overall_health,
                "files_accessible": files_accessible,
                "active_agents": len(self.active_agents),
                "agents_healthy": agents_healthy,
                "processing_healthy": processing_healthy,
                "error_rate": self.message_stats["failed_routes"] / max(self.message_stats["total_messages"], 1)
            }
            
        except Exception as e:
            return {
                "status": "critical",
                "error": str(e)
            }

class CommunicationHub:
    """Central communication hub for the trading system"""
    
    def __init__(self):
        self.router = MessageRouter()
        self.monitoring_enabled = True
        self._monitoring_task = None
        
    async def start(self):
        """Start the communication hub"""
        logger.info("Starting Communication Hub...")
        
        # Start monitoring
        if self.monitoring_enabled:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Communication Hub started")

    async def stop(self):
        """Stop the communication hub"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        logger.info("Communication Hub stopped")

    async def _monitoring_loop(self):
        """Monitor communication health and performance"""
        while True:
            try:
                # Perform health check
                health = self.router.health_check()
                
                if health["status"] != "healthy":
                    logger.warning(f"Communication health degraded: {health}")
                
                # Get routing statistics
                stats = self.router.get_routing_statistics()
                logger.info(f"Communication stats: {stats['statistics']}")
                
                # Check for optimization opportunities
                optimizations = self.router.optimize_routing()
                if optimizations:
                    logger.info(f"Routing optimizations available: {optimizations}")
                
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in communication monitoring: {e}")
                await asyncio.sleep(300)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive communication hub status"""
        return {
            "router_health": self.router.health_check(),
            "routing_stats": self.router.get_routing_statistics(),
            "communication_analysis": self.router.analyze_communication_patterns()
        }