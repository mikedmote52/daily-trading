#!/usr/bin/env python3
"""
Master Orchestration Agent

Coordinates all specialized agents in the stock trading system.
Handles system-wide communication, monitoring, and optimization.
"""

import asyncio
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import redis
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MasterAgent')

@dataclass
class AgentStatus:
    name: str
    status: str  # 'online', 'offline', 'busy'
    last_heartbeat: datetime
    current_task: Optional[str] = None
    performance_metrics: Dict[str, Any] = None

@dataclass
class SystemMetrics:
    total_agents: int
    active_agents: int
    total_trades: int
    portfolio_value: float
    daily_pnl: float
    system_health: str

class MasterOrchestrationAgent:
    def __init__(self):
        self.claude = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
        self.redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
        self.agents: Dict[str, AgentStatus] = {}
        self.system_prompt = self._load_system_prompt()
        self.running = False
        
    def _load_system_prompt(self) -> str:
        return """You are the Master Orchestration Agent for an AI-powered stock trading system.

Your responsibilities:
1. Coordinate specialized agents (frontend, backend, discovery, backtesting, portfolio)
2. Monitor system performance and health
3. Optimize resource allocation and task distribution
4. Ensure smooth integration between all components
5. Identify bottlenecks and improvement opportunities
6. Facilitate communication between agents and stakeholders

Core Principles:
- Maintain system stability and reliability
- Optimize for performance and profitability
- Ensure proper risk management
- Facilitate data-driven decision making
- Maintain clear audit trails

Communication Protocol:
- Use Redis pub/sub for inter-agent messaging
- Maintain agent registry with health status
- Coordinate task scheduling and resource allocation
- Monitor and report system-wide metrics"""

    async def start(self):
        """Start the master orchestration system"""
        logger.info("Starting Master Orchestration Agent...")
        self.running = True
        
        # Initialize agent registry
        await self._initialize_agent_registry()
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._monitor_agents()),
            asyncio.create_task(self._system_health_check()),
            asyncio.create_task(self._message_handler()),
            asyncio.create_task(self._optimization_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down Master Orchestration Agent...")
            self.running = False

    async def _initialize_agent_registry(self):
        """Initialize the agent registry"""
        agent_names = ['frontend', 'backend', 'discovery', 'backtesting', 'portfolio']
        
        for name in agent_names:
            self.agents[name] = AgentStatus(
                name=name,
                status='offline',
                last_heartbeat=datetime.now(),
                performance_metrics={}
            )
            
        # Store in Redis
        await self._update_agent_registry()

    async def _update_agent_registry(self):
        """Update agent registry in Redis"""
        registry_data = {name: asdict(status) for name, status in self.agents.items()}
        self.redis_client.set('agent_registry', json.dumps(registry_data, default=str))

    async def _monitor_agents(self):
        """Monitor agent health and status"""
        while self.running:
            try:
                # Check agent heartbeats
                for agent_name, agent_status in self.agents.items():
                    last_heartbeat = self.redis_client.get(f'heartbeat:{agent_name}')
                    
                    if last_heartbeat:
                        agent_status.last_heartbeat = datetime.fromisoformat(
                            last_heartbeat.decode()
                        )
                        
                        # Check if agent is responsive (within last 30 seconds)
                        if (datetime.now() - agent_status.last_heartbeat).seconds < 30:
                            agent_status.status = 'online'
                        else:
                            agent_status.status = 'offline'
                            logger.warning(f"Agent {agent_name} appears offline")
                
                await self._update_agent_registry()
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in agent monitoring: {e}")
                await asyncio.sleep(10)

    async def _system_health_check(self):
        """Perform system-wide health checks"""
        while self.running:
            try:
                # Calculate system metrics
                active_agents = sum(1 for a in self.agents.values() if a.status == 'online')
                
                metrics = SystemMetrics(
                    total_agents=len(self.agents),
                    active_agents=active_agents,
                    total_trades=0,  # Will be updated by portfolio agent
                    portfolio_value=0.0,  # Will be updated by portfolio agent
                    daily_pnl=0.0,  # Will be updated by portfolio agent
                    system_health='healthy' if active_agents >= 3 else 'degraded'
                )
                
                # Store system metrics
                self.redis_client.set('system_metrics', json.dumps(asdict(metrics), default=str))
                
                # Log system status
                logger.info(f"System Health: {metrics.system_health} - "
                          f"{active_agents}/{len(self.agents)} agents online")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in system health check: {e}")
                await asyncio.sleep(60)

    async def _message_handler(self):
        """Handle inter-agent messages"""
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe('master_channel')
        
        while self.running:
            try:
                message = pubsub.get_message(timeout=1)
                if message and message['type'] == 'message':
                    await self._process_message(json.loads(message['data']))
                    
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
                await asyncio.sleep(1)

    async def _process_message(self, message: Dict[str, Any]):
        """Process incoming messages from agents"""
        msg_type = message.get('type')
        sender = message.get('sender')
        data = message.get('data', {})
        
        logger.info(f"Received message from {sender}: {msg_type}")
        
        if msg_type == 'heartbeat':
            # Update agent heartbeat
            if sender in self.agents:
                self.agents[sender].last_heartbeat = datetime.now()
                
        elif msg_type == 'task_request':
            # Handle task requests
            await self._handle_task_request(sender, data)
            
        elif msg_type == 'status_update':
            # Update agent status
            if sender in self.agents:
                self.agents[sender].current_task = data.get('current_task')
                self.agents[sender].performance_metrics = data.get('metrics', {})
                
        elif msg_type == 'alert':
            # Handle system alerts
            logger.warning(f"Alert from {sender}: {data.get('message')}")

    async def _handle_task_request(self, sender: str, request_data: Dict[str, Any]):
        """Handle task requests from agents"""
        task_type = request_data.get('task_type')
        
        # Use Claude to determine optimal task routing
        prompt = f"""
        Agent {sender} is requesting assistance with: {task_type}
        Request details: {json.dumps(request_data, indent=2)}
        
        Current agent status:
        {json.dumps({name: asdict(status) for name, status in self.agents.items()}, indent=2, default=str)}
        
        Please recommend:
        1. Which agent(s) should handle this task
        2. Task priority (high, medium, low)
        3. Any coordination needed between agents
        4. Potential risks or considerations
        
        Provide a JSON response with your recommendations.
        """
        
        try:
            response = self.claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse Claude's response and route the task
            # This is a simplified version - in production, you'd have more sophisticated routing
            logger.info(f"Claude recommendation for task from {sender}: {response.content}")
            
        except Exception as e:
            logger.error(f"Error getting Claude recommendation: {e}")

    async def _optimization_loop(self):
        """Continuous system optimization"""
        while self.running:
            try:
                # Collect system-wide performance data
                performance_data = {}
                for agent_name in self.agents:
                    metrics = self.redis_client.get(f'metrics:{agent_name}')
                    if metrics:
                        performance_data[agent_name] = json.loads(metrics)
                
                # Use Claude to analyze and suggest optimizations
                if performance_data:
                    await self._analyze_system_performance(performance_data)
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(300)

    async def _analyze_system_performance(self, performance_data: Dict[str, Any]):
        """Analyze system performance and suggest optimizations"""
        prompt = f"""
        Analyze the following system performance data and suggest optimizations:
        
        {json.dumps(performance_data, indent=2, default=str)}
        
        Consider:
        1. Resource utilization efficiency
        2. Bottlenecks in the pipeline
        3. Agent coordination improvements
        4. Risk management enhancements
        5. Performance optimization opportunities
        
        Provide specific, actionable recommendations.
        """
        
        try:
            response = self.claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            logger.info(f"System optimization recommendations: {response.content}")
            
            # Store recommendations for other agents to access
            self.redis_client.set(
                'optimization_recommendations', 
                response.content[0].text if response.content else ""
            )
            
        except Exception as e:
            logger.error(f"Error analyzing system performance: {e}")

    def send_message_to_agent(self, agent_name: str, message: Dict[str, Any]):
        """Send message to specific agent"""
        self.redis_client.publish(f'{agent_name}_channel', json.dumps(message))

    def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all agents"""
        for agent_name in self.agents:
            self.send_message_to_agent(agent_name, message)

async def main():
    """Main entry point"""
    master = MasterOrchestrationAgent()
    
    try:
        await master.start()
    except Exception as e:
        logger.error(f"Fatal error in master agent: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())