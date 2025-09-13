#!/usr/bin/env python3
"""
Enhanced Master Orchestration Agent with Claude Code SDK Integration

Coordinates all specialized agents in the stock trading system with
enhanced communication and shared context management.
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
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EnhancedMasterAgent')

class EnhancedMasterOrchestrationAgent:
    def __init__(self):
        # Initialize Claude SDK messenger
        self.messenger = ClaudeSDKMessenger(
            agent_name="master_orchestration",
            shared_context_dir="../../../shared_context"
        )
        
        # Initialize Claude AI
        self.claude = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
        
        # Agent coordination state
        self.coordination_tasks = []
        self.system_optimization_history = []
        self.running = False
        
        # Register message handlers
        self._register_message_handlers()
        
        logger.info("Enhanced Master Orchestration Agent initialized")

    def _register_message_handlers(self):
        """Register handlers for different message types"""
        
        def handle_agent_status_update(message):
            """Handle status updates from other agents"""
            sender = message["sender"]
            content = message["content"]
            
            logger.info(f"Received status update from {sender}: {content}")
            
            # Update shared progress with agent status
            self.messenger.update_progress(
                status="online",
                current_task=f"Processing status from {sender}",
                progress_data={"last_coordination": datetime.now().isoformat()}
            )
            
            # Check if coordination is needed
            asyncio.create_task(self._evaluate_coordination_needs(sender, content))
        
        def handle_system_alert(message):
            """Handle system alerts from any agent"""
            content = message["content"]
            priority = message.get("priority", "normal")
            
            logger.warning(f"System alert received: {content}")
            
            # Add to system alerts
            self.messenger.add_alert("warning", content, "system")
            
            # If high priority, take immediate action
            if priority == "high":
                asyncio.create_task(self._handle_critical_alert(content))
        
        def handle_optimization_request(message):
            """Handle requests for system optimization"""
            content = message["content"]
            sender = message["sender"]
            
            logger.info(f"Optimization request from {sender}: {content}")
            
            # Queue optimization task
            asyncio.create_task(self._perform_system_optimization(content))
        
        def handle_coordination_request(message):
            """Handle requests for agent coordination"""
            content = message["content"]
            sender = message["sender"]
            
            logger.info(f"Coordination request from {sender}")
            
            asyncio.create_task(self._coordinate_agents(content))
        
        # Register handlers
        self.messenger.register_message_handler("system_status", handle_agent_status_update)
        self.messenger.register_message_handler("system_alert", handle_system_alert)
        self.messenger.register_message_handler("optimization_request", handle_optimization_request)
        self.messenger.register_message_handler("coordination_request", handle_coordination_request)

    async def start(self):
        """Start the enhanced master orchestration system"""
        logger.info("Starting Enhanced Master Orchestration Agent...")
        self.running = True
        
        # Update initial status
        self.messenger.update_progress(
            status="online",
            current_task="Starting coordination system",
            progress_data={
                "startup_time": datetime.now().isoformat(),
                "coordination_tasks": 0,
                "optimizations_performed": 0
            }
        )
        
        # Start message listening
        self.messenger.start_listening(poll_interval=1.0)
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._system_monitoring_loop()),
            asyncio.create_task(self._coordination_loop()),
            asyncio.create_task(self._optimization_loop()),
            asyncio.create_task(self._health_reporting_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down Enhanced Master Orchestration Agent...")
            self.running = False
            self.messenger.stop_listening()

    async def _system_monitoring_loop(self):
        """Monitor overall system health and performance"""
        while self.running:
            try:
                # Get system health
                health = self.messenger.get_system_health()
                
                # Get all agent progress
                progress = self.messenger.get_progress()
                
                # Update system metrics
                active_agents = health["active_agents"]
                total_agents = health["total_agents"]
                
                # Update shared data
                self.messenger.update_shared_data("system_health", health)
                
                # Log system status
                logger.info(f"System Health: {health['health']} - "
                          f"{active_agents}/{total_agents} agents active")
                
                # Check for system issues
                if health["health"] in ["critical", "degraded"]:
                    await self._handle_system_degradation(health)
                
                # Update progress
                self.messenger.update_progress(
                    status="online",
                    current_task="Monitoring system health",
                    progress_data={
                        "last_health_check": datetime.now().isoformat(),
                        "system_health": health["health"],
                        "active_agents": active_agents
                    },
                    performance_metrics={
                        "system_uptime": self._calculate_uptime(),
                        "coordination_efficiency": self._calculate_coordination_efficiency()
                    }
                )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(30)

    async def _coordination_loop(self):
        """Main coordination loop for inter-agent tasks"""
        while self.running:
            try:
                # Get messages and process coordination needs
                messages = self.messenger.get_messages(mark_as_read=False)
                
                # Look for coordination opportunities
                coordination_needed = self._identify_coordination_opportunities(messages)
                
                if coordination_needed:
                    await self._execute_coordination_tasks(coordination_needed)
                
                # Process any queued coordination tasks
                if self.coordination_tasks:
                    await self._process_coordination_queue()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(60)

    async def _optimization_loop(self):
        """Perform system-wide optimizations"""
        while self.running:
            try:
                # Wait 5 minutes before first optimization
                await asyncio.sleep(300)
                
                logger.info("Running system optimization analysis...")
                
                # Get performance data from all agents
                progress_data = self.messenger.get_progress()
                
                if not progress_data.get("agents"):
                    await asyncio.sleep(300)
                    continue
                
                # Use Claude to analyze system performance
                optimization_suggestions = await self._analyze_system_performance(progress_data)
                
                if optimization_suggestions:
                    # Store optimization suggestions
                    self.messenger.update_shared_data("optimization_suggestions", optimization_suggestions)
                    
                    # Send optimization recommendations to relevant agents
                    await self._distribute_optimization_recommendations(optimization_suggestions)
                
                self.messenger.update_progress(
                    status="online",
                    progress_data={
                        "last_optimization": datetime.now().isoformat(),
                        "optimizations_performed": len(self.system_optimization_history)
                    }
                )
                
                await asyncio.sleep(1800)  # Run every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(1800)

    async def _health_reporting_loop(self):
        """Send periodic health reports and system updates"""
        while self.running:
            try:
                # Collect comprehensive system status
                system_report = await self._generate_system_report()
                
                # Broadcast system status to all agents
                self.messenger.broadcast_message(
                    "system_updates",
                    system_report,
                    priority="normal"
                )
                
                # Send detailed status to frontend
                self.messenger.send_message(
                    "frontend_interface",
                    "system_status",
                    system_report
                )
                
                await asyncio.sleep(120)  # Report every 2 minutes
                
            except Exception as e:
                logger.error(f"Error in health reporting: {e}")
                await asyncio.sleep(120)

    async def _evaluate_coordination_needs(self, sender: str, content: Dict[str, Any]):
        """Evaluate if coordination is needed based on agent status"""
        try:
            # Check if this agent needs resources from others
            if "resource_request" in content:
                await self._handle_resource_request(sender, content["resource_request"])
            
            # Check for performance issues requiring intervention
            if content.get("performance_metrics", {}).get("error_rate", 0) > 0.05:
                await self._handle_performance_issue(sender, content)
            
            # Check for task dependencies
            if "dependencies" in content:
                await self._coordinate_dependent_tasks(sender, content["dependencies"])
                
        except Exception as e:
            logger.error(f"Error evaluating coordination needs: {e}")

    async def _handle_critical_alert(self, alert_content: str):
        """Handle critical system alerts"""
        try:
            logger.critical(f"Handling critical alert: {alert_content}")
            
            # Add to high-priority alerts
            self.messenger.add_alert("critical", f"CRITICAL: {alert_content}", "system")
            
            # Analyze alert with Claude for recommended actions
            prompt = f"""
            Critical system alert received: {alert_content}
            
            Please provide immediate action recommendations:
            1. Severity assessment (1-10)
            2. Immediate actions needed
            3. Agents that need notification
            4. Risk mitigation steps
            5. Recovery procedures
            
            Respond with a JSON object containing these recommendations.
            """
            
            response = self.claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse and execute recommendations
            import json
            recommendations = json.loads(response.content[0].text)
            
            # Execute immediate actions
            await self._execute_emergency_procedures(recommendations)
            
        except Exception as e:
            logger.error(f"Error handling critical alert: {e}")

    async def _analyze_system_performance(self, progress_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze system performance using Claude and provide optimization suggestions"""
        try:
            # Prepare performance summary for Claude
            performance_summary = {
                "system_health": self.messenger.get_system_health(),
                "agent_statuses": progress_data.get("agents", {}),
                "communication_stats": len(progress_data.get("communication_log", [])),
                "shared_data": progress_data.get("shared_data", {})
            }
            
            prompt = f"""
            Analyze the following trading system performance data and provide optimization recommendations:
            
            {json.dumps(performance_summary, indent=2, default=str)}
            
            Please provide:
            1. Performance bottlenecks identified
            2. Agent coordination improvements
            3. Resource allocation optimizations
            4. Risk management enhancements
            5. System efficiency improvements
            
            Focus on actionable recommendations that can improve:
            - System reliability and uptime
            - Inter-agent communication efficiency
            - Trading strategy performance
            - Risk-adjusted returns
            
            Respond with a JSON object containing specific recommendations.
            """
            
            response = self.claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            recommendations = json.loads(response.content[0].text)
            
            # Store in optimization history
            self.system_optimization_history.append({
                "timestamp": datetime.now().isoformat(),
                "recommendations": recommendations
            })
            
            logger.info("System performance analysis completed")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing system performance: {e}")
            return None

    async def _distribute_optimization_recommendations(self, recommendations: Dict[str, Any]):
        """Send optimization recommendations to relevant agents"""
        try:
            # Send specific recommendations to each agent
            agent_recommendations = {
                "stock_discovery": recommendations.get("discovery_optimizations", []),
                "backtesting": recommendations.get("backtesting_optimizations", []),
                "portfolio_management": recommendations.get("portfolio_optimizations", []),
                "backend_orchestration": recommendations.get("backend_optimizations", [])
            }
            
            for agent, recs in agent_recommendations.items():
                if recs:
                    self.messenger.send_message(
                        agent,
                        "optimization_suggestion",
                        {
                            "recommendations": recs,
                            "priority": "normal",
                            "source": "master_coordination"
                        }
                    )
            
        except Exception as e:
            logger.error(f"Error distributing optimization recommendations: {e}")

    def _identify_coordination_opportunities(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify opportunities for agent coordination"""
        coordination_needed = []
        
        # Look for agents requesting similar resources
        # Look for complementary tasks that could be coordinated
        # Look for performance issues that could be resolved through coordination
        
        return coordination_needed

    async def _generate_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system status report"""
        health = self.messenger.get_system_health()
        progress = self.messenger.get_progress()
        shared_data = self.messenger.get_shared_data()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": health,
            "agent_count": health["total_agents"],
            "active_agents": health["active_agents"],
            "system_uptime": self._calculate_uptime(),
            "coordination_efficiency": self._calculate_coordination_efficiency(),
            "optimization_count": len(self.system_optimization_history),
            "last_optimization": self.system_optimization_history[-1]["timestamp"] if self.system_optimization_history else None,
            "portfolio_summary": shared_data.get("portfolio_summary", {}),
            "market_status": shared_data.get("market_status", "unknown")
        }

    def _calculate_uptime(self) -> float:
        """Calculate system uptime in hours"""
        # Simplified uptime calculation
        return 1.0  # Would be calculated from startup time

    def _calculate_coordination_efficiency(self) -> float:
        """Calculate coordination efficiency score"""
        # Simplified efficiency calculation
        return 0.95  # Would be calculated from successful coordinations

    # Additional helper methods...
    async def _handle_system_degradation(self, health: Dict[str, Any]):
        """Handle system degradation events"""
        pass

    async def _handle_resource_request(self, sender: str, request: Dict[str, Any]):
        """Handle resource requests from agents"""
        pass

    async def _handle_performance_issue(self, sender: str, content: Dict[str, Any]):
        """Handle performance issues from agents"""
        pass

    async def _coordinate_dependent_tasks(self, sender: str, dependencies: List[str]):
        """Coordinate tasks with dependencies"""
        pass

    async def _execute_coordination_tasks(self, tasks: List[Dict[str, Any]]):
        """Execute coordination tasks"""
        pass

    async def _process_coordination_queue(self):
        """Process queued coordination tasks"""
        pass

    async def _execute_emergency_procedures(self, recommendations: Dict[str, Any]):
        """Execute emergency procedures based on recommendations"""
        pass

async def main():
    """Main entry point"""
    master = EnhancedMasterOrchestrationAgent()
    
    try:
        await master.start()
    except Exception as e:
        logger.error(f"Fatal error in enhanced master agent: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())