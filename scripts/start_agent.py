#!/usr/bin/env python3
"""
Agent Starter Script

Utility to start individual agents with proper environment setup.
"""

import os
import sys
import subprocess
import argparse
import signal
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def start_agent(agent_name: str):
    """Start a specific agent"""
    agent_dir = PROJECT_ROOT / "agents" / agent_name
    venv_python = agent_dir / "venv" / "bin" / "python"
    main_script = agent_dir / "main.py"
    
    if not agent_dir.exists():
        print(f"❌ Agent directory not found: {agent_dir}")
        return False
    
    if not venv_python.exists():
        print(f"❌ Virtual environment not found for {agent_name}")
        print(f"   Run setup.sh first or create venv manually")
        return False
    
    if not main_script.exists():
        print(f"❌ Main script not found: {main_script}")
        return False
    
    print(f"🚀 Starting {agent_name} agent...")
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    
    try:
        # Start the agent
        process = subprocess.Popen(
            [str(venv_python), str(main_script)],
            cwd=str(agent_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print(f"✅ {agent_name} agent started (PID: {process.pid})")
        
        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print(f"\n🛑 Stopping {agent_name} agent...")
            process.terminate()
            process.wait()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Stream output
        for line in process.stdout:
            print(line.rstrip())
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to start {agent_name} agent: {e}")
        return False

def start_frontend():
    """Start the frontend React app"""
    frontend_dir = PROJECT_ROOT / "agents" / "frontend"
    
    if not frontend_dir.exists():
        print(f"❌ Frontend directory not found: {frontend_dir}")
        return False
    
    print("🚀 Starting frontend agent...")
    
    try:
        process = subprocess.Popen(
            ["npm", "start"],
            cwd=str(frontend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print(f"✅ Frontend agent started (PID: {process.pid})")
        
        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print("\n🛑 Stopping frontend agent...")
            process.terminate()
            process.wait()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Stream output
        for line in process.stdout:
            print(line.rstrip())
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to start frontend agent: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Start trading system agents")
    parser.add_argument(
        "agent",
        choices=["master", "frontend", "backend", "discovery", "backtesting", "portfolio"],
        help="Agent to start"
    )
    
    args = parser.parse_args()
    
    if args.agent == "frontend":
        success = start_frontend()
    else:
        success = start_agent(args.agent)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()