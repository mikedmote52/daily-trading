#!/usr/bin/env python3
"""
Enhanced System Starter with Claude Code SDK Integration

Starts all agents with proper inter-agent communication and shared context.
"""

import os
import sys
import subprocess
import asyncio
import signal
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
import time

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SystemStarter')

class EnhancedSystemStarter:
    """Enhanced system starter with communication management"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.agent_configs = self._load_agent_configs()
        self.running = False
        
    def _load_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load agent configurations"""
        configs = {}
        config_dir = PROJECT_ROOT / ".claude" / "agents"
        
        if config_dir.exists():
            for agent_dir in config_dir.iterdir():
                if agent_dir.is_dir():
                    config_file = agent_dir / f"{agent_dir.name.replace('_', '_')}_agent.json"
                    if not config_file.exists():
                        # Try alternative naming
                        config_files = list(agent_dir.glob("*.json"))
                        if config_files:
                            config_file = config_files[0]
                    
                    if config_file.exists():
                        try:
                            with open(config_file, 'r') as f:
                                config = json.load(f)
                                configs[config["name"]] = config
                        except Exception as e:
                            logger.warning(f"Failed to load config for {agent_dir.name}: {e}")
        
        return configs

    async def start_system(self, agents: List[str] = None):
        """Start the enhanced trading system"""
        logger.info("🚀 Starting Enhanced Daily Trading System with Claude Code SDK...")
        
        # Initialize shared context
        await self._initialize_shared_context()
        
        # Start communication hub
        await self._start_communication_hub()
        
        # Determine which agents to start
        if agents is None:
            agents = ["master_orchestration", "frontend_interface", "backend_orchestration", 
                     "stock_discovery", "backtesting", "portfolio_management"]
        
        # Start agents in dependency order
        startup_order = self._get_startup_order(agents)
        
        for agent_name in startup_order:
            success = await self._start_agent(agent_name)
            if not success:
                logger.error(f"Failed to start {agent_name}, continuing with other agents...")
            else:
                # Wait a bit between agent starts
                await asyncio.sleep(2)
        
        self.running = True
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Monitor system
        await self._monitor_system()

    async def _initialize_shared_context(self):
        """Initialize shared context files"""
        try:
            shared_context_dir = PROJECT_ROOT / "shared_context"
            shared_context_dir.mkdir(exist_ok=True)
            
            # Initialize progress.json if it doesn't exist
            progress_file = shared_context_dir / "progress.json"
            if not progress_file.exists():
                initial_progress = {
                    "system_status": {
                        "startup_time": None,
                        "last_update": None,
                        "total_agents": 6,
                        "active_agents": 0,
                        "system_health": "initializing"
                    },
                    "agents": {},
                    "communication_log": [],
                    "shared_data": {
                        "market_status": "closed",
                        "system_alerts": [],
                        "portfolio_summary": {
                            "total_value": 100000,
                            "daily_pnl": 0,
                            "positions_count": 0
                        }
                    }
                }
                
                with open(progress_file, 'w') as f:
                    json.dump(initial_progress, f, indent=2, default=str)
            
            # Initialize messages.json if it doesn't exist
            messages_file = shared_context_dir / "messages.json"
            if not messages_file.exists():
                initial_messages = {
                    "message_queue": [],
                    "message_history": [],
                    "agent_subscriptions": {
                        "master_orchestration": ["all"],
                        "frontend_interface": ["system_updates", "portfolio_updates", "alerts"],
                        "backend_orchestration": ["trade_requests", "data_updates", "system_status"],
                        "stock_discovery": ["screening_requests", "market_updates"],
                        "backtesting": ["backtest_requests", "strategy_updates"],
                        "portfolio_management": ["trade_signals", "risk_updates", "rebalance_requests"]
                    }
                }
                
                with open(messages_file, 'w') as f:
                    json.dump(initial_messages, f, indent=2)
            
            logger.info("✅ Shared context initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize shared context: {e}")

    async def _start_communication_hub(self):
        """Start the communication hub"""
        try:
            # The communication hub is integrated into the Claude SDK helper
            # Just verify it's working
            from shared.communication.message_router import CommunicationHub
            
            self.comm_hub = CommunicationHub()
            await self.comm_hub.start()
            
            logger.info("✅ Communication hub started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start communication hub: {e}")

    def _get_startup_order(self, agents: List[str]) -> List[str]:
        """Get optimal startup order for agents"""
        # Define dependency order
        order_priority = {
            "master_orchestration": 1,
            "backend_orchestration": 2,
            "stock_discovery": 3,
            "backtesting": 3,
            "portfolio_management": 4,
            "frontend_interface": 5
        }
        
        return sorted(agents, key=lambda x: order_priority.get(x, 999))

    async def _start_agent(self, agent_name: str) -> bool:
        """Start a specific agent"""
        try:
            logger.info(f"🔧 Starting {agent_name} agent...")
            
            # Determine agent directory and script
            agent_dir = PROJECT_ROOT / "agents" / agent_name.replace("_orchestration", "").replace("_interface", "").replace("_management", "")
            
            # Check for enhanced version first
            enhanced_script = agent_dir / "enhanced_main.py"
            main_script = agent_dir / "main.py"
            
            if enhanced_script.exists():
                script_path = enhanced_script
                logger.info(f"   Using enhanced version for {agent_name}")
            elif main_script.exists():
                script_path = main_script
                logger.info(f"   Using standard version for {agent_name}")
            else:
                logger.error(f"   No main script found for {agent_name}")
                return False
            
            # Handle frontend separately (React app)
            if agent_name == "frontend_interface":
                return await self._start_frontend_agent()
            
            # Start Python agent
            venv_python = agent_dir / "venv" / "bin" / "python"
            if not venv_python.exists():
                logger.warning(f"   No virtual environment found for {agent_name}, using system Python")
                venv_python = "python3"
            
            # Set environment variables
            env = os.environ.copy()
            env['PYTHONPATH'] = str(PROJECT_ROOT)
            
            # Start the process
            process = subprocess.Popen(
                [str(venv_python), str(script_path)],
                cwd=str(agent_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes[agent_name] = process
            
            # Wait a moment and check if process started successfully
            await asyncio.sleep(1)
            if process.poll() is None:
                logger.info(f"✅ {agent_name} agent started (PID: {process.pid})")
                return True
            else:
                logger.error(f"❌ {agent_name} agent failed to start")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start {agent_name} agent: {e}")
            return False

    async def _start_frontend_agent(self) -> bool:
        """Start the frontend React app"""
        try:
            frontend_dir = PROJECT_ROOT / "agents" / "frontend"
            
            if not (frontend_dir / "package.json").exists():
                logger.error("Frontend package.json not found")
                return False
            
            process = subprocess.Popen(
                ["npm", "start"],
                cwd=str(frontend_dir),
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes["frontend_interface"] = process
            
            # Wait a bit longer for React to start
            await asyncio.sleep(3)
            if process.poll() is None:
                logger.info(f"✅ Frontend agent started (PID: {process.pid})")
                return True
            else:
                logger.error("❌ Frontend agent failed to start")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start frontend agent: {e}")
            return False

    async def _monitor_system(self):
        """Monitor the running system"""
        logger.info("📊 System monitoring started...")
        
        while self.running:
            try:
                # Check process health
                dead_agents = []
                for agent_name, process in self.processes.items():
                    if process.poll() is not None:
                        dead_agents.append(agent_name)
                
                # Report dead agents
                for agent_name in dead_agents:
                    logger.warning(f"⚠️  Agent {agent_name} has stopped")
                    del self.processes[agent_name]
                
                # System status
                active_count = len(self.processes)
                logger.info(f"📈 System Status: {active_count} agents running")
                
                # Check communication hub
                if hasattr(self, 'comm_hub'):
                    hub_status = self.comm_hub.get_status()
                    logger.info(f"📡 Communication: {hub_status['router_health']['status']}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(30)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Received signal {signum}, shutting down system...")
        self.running = False
        asyncio.create_task(self._shutdown_system())

    async def _shutdown_system(self):
        """Gracefully shutdown the system"""
        logger.info("🔄 Shutting down Enhanced Daily Trading System...")
        
        # Stop communication hub
        if hasattr(self, 'comm_hub'):
            await self.comm_hub.stop()
        
        # Terminate all processes
        for agent_name, process in self.processes.items():
            logger.info(f"   Stopping {agent_name}...")
            try:
                process.terminate()
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            except Exception as e:
                logger.error(f"Error stopping {agent_name}: {e}")
        
        logger.info("✅ System shutdown complete")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "running": self.running,
            "active_agents": len(self.processes),
            "agents": {
                name: {
                    "pid": process.pid,
                    "status": "running" if process.poll() is None else "stopped"
                }
                for name, process in self.processes.items()
            }
        }

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Enhanced Daily Trading System")
    parser.add_argument(
        "--agents",
        nargs='+',
        choices=["master_orchestration", "frontend_interface", "backend_orchestration", 
                "stock_discovery", "backtesting", "portfolio_management"],
        help="Specific agents to start (default: all)"
    )
    
    args = parser.parse_args()
    
    starter = EnhancedSystemStarter()
    
    try:
        await starter.start_system(agents=args.agents)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())