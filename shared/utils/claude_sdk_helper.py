#!/usr/bin/env python3
"""
Claude Code SDK Helper

Utilities for inter-agent communication using Claude Code SDK patterns.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import time

logger = logging.getLogger(__name__)

class ClaudeSDKMessenger:
    """Enhanced messaging system for Claude Code SDK agent communication"""
    
    def __init__(self, agent_name: str, shared_context_dir: str = "shared_context"):
        self.agent_name = agent_name
        self.shared_context_dir = Path(shared_context_dir)
        self.progress_file = self.shared_context_dir / "progress.json"
        self.messages_file = self.shared_context_dir / "messages.json"
        self._lock = threading.Lock()
        self._message_handlers = {}
        self._listening = False
        
        # Ensure shared context directory exists
        self.shared_context_dir.mkdir(exist_ok=True)
        
        # Initialize files if they don't exist
        self._ensure_files_exist()
    
    def _ensure_files_exist(self):
        """Ensure required files exist with default structure"""
        if not self.progress_file.exists():
            default_progress = {
                "system_status": {"startup_time": None, "last_update": None},
                "agents": {},
                "communication_log": [],
                "shared_data": {}
            }
            self._write_file(self.progress_file, default_progress)
        
        if not self.messages_file.exists():
            default_messages = {
                "message_queue": [],
                "message_history": [],
                "agent_subscriptions": {},
                "message_types": {}
            }
            self._write_file(self.messages_file, default_messages)
    
    def _read_file(self, file_path: Path) -> Dict[str, Any]:
        """Safely read JSON file with error handling"""
        try:
            with self._lock:
                with open(file_path, 'r') as f:
                    return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error reading {file_path}: {e}")
            return {}
    
    def _write_file(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """Safely write JSON file with error handling"""
        try:
            with self._lock:
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Error writing {file_path}: {e}")
            return False
    
    def update_progress(self, status: str, current_task: str = None, 
                       progress_data: Dict[str, Any] = None, 
                       performance_metrics: Dict[str, Any] = None) -> bool:
        """Update agent progress in shared context"""
        try:
            data = self._read_file(self.progress_file)
            
            # Initialize agent data if not exists
            if self.agent_name not in data.get("agents", {}):
                data.setdefault("agents", {})[self.agent_name] = {
                    "status": "offline",
                    "last_heartbeat": None,
                    "current_task": None,
                    "progress": {},
                    "performance_metrics": {}
                }
            
            # Update agent data
            agent_data = data["agents"][self.agent_name]
            agent_data["status"] = status
            agent_data["last_heartbeat"] = datetime.now().isoformat()
            
            if current_task:
                agent_data["current_task"] = current_task
            
            if progress_data:
                agent_data["progress"].update(progress_data)
            
            if performance_metrics:
                agent_data["performance_metrics"].update(performance_metrics)
            
            # Update system status
            data["system_status"]["last_update"] = datetime.now().isoformat()
            active_agents = sum(1 for agent in data["agents"].values() 
                              if agent["status"] in ["online", "busy"])
            data["system_status"]["active_agents"] = active_agents
            
            return self._write_file(self.progress_file, data)
            
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
            return False
    
    def get_progress(self, agent_name: str = None) -> Dict[str, Any]:
        """Get progress data for specific agent or all agents"""
        data = self._read_file(self.progress_file)
        
        if agent_name:
            return data.get("agents", {}).get(agent_name, {})
        return data
    
    def send_message(self, recipient: str, message_type: str, content: Any,
                    priority: str = "normal") -> bool:
        """Send message to another agent"""
        try:
            messages_data = self._read_file(self.messages_file)
            
            message = {
                "id": f"{self.agent_name}_{int(time.time() * 1000)}",
                "sender": self.agent_name,
                "recipient": recipient,
                "message_type": message_type,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "priority": priority,
                "status": "pending"
            }
            
            # Add to message queue
            messages_data.setdefault("message_queue", []).append(message)
            
            # Also log communication
            progress_data = self._read_file(self.progress_file)
            progress_data.setdefault("communication_log", []).append({
                "from": self.agent_name,
                "to": recipient,
                "type": message_type,
                "timestamp": message["timestamp"]
            })
            
            # Keep only last 100 log entries
            progress_data["communication_log"] = progress_data["communication_log"][-100:]
            
            # Write both files
            success1 = self._write_file(self.messages_file, messages_data)
            success2 = self._write_file(self.progress_file, progress_data)
            
            logger.info(f"Message sent from {self.agent_name} to {recipient}: {message_type}")
            return success1 and success2
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def get_messages(self, mark_as_read: bool = True) -> List[Dict[str, Any]]:
        """Get messages for this agent"""
        try:
            messages_data = self._read_file(self.messages_file)
            queue = messages_data.get("message_queue", [])
            
            # Filter messages for this agent
            my_messages = [msg for msg in queue 
                          if msg["recipient"] == self.agent_name or msg["recipient"] == "all"]
            
            if mark_as_read and my_messages:
                # Remove processed messages and move to history
                remaining_queue = [msg for msg in queue 
                                 if msg["recipient"] != self.agent_name and msg["recipient"] != "all"]
                
                # Update message status to read
                for msg in my_messages:
                    msg["status"] = "read"
                    msg["read_timestamp"] = datetime.now().isoformat()
                
                # Move to history and keep last 500
                messages_data.setdefault("message_history", []).extend(my_messages)
                messages_data["message_history"] = messages_data["message_history"][-500:]
                
                # Update queue
                messages_data["message_queue"] = remaining_queue
                
                self._write_file(self.messages_file, messages_data)
            
            return my_messages
            
        except Exception as e:
            logger.error(f"Error getting messages: {e}")
            return []
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register handler for specific message type"""
        self._message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    def start_listening(self, poll_interval: float = 1.0):
        """Start listening for messages in background thread"""
        if self._listening:
            return
        
        self._listening = True
        
        def listen_loop():
            while self._listening:
                try:
                    messages = self.get_messages()
                    
                    for message in messages:
                        message_type = message.get("message_type")
                        if message_type in self._message_handlers:
                            try:
                                self._message_handlers[message_type](message)
                            except Exception as e:
                                logger.error(f"Error processing message {message['id']}: {e}")
                    
                    time.sleep(poll_interval)
                    
                except Exception as e:
                    logger.error(f"Error in message listening loop: {e}")
                    time.sleep(poll_interval)
        
        listen_thread = threading.Thread(target=listen_loop, daemon=True)
        listen_thread.start()
        logger.info(f"Started message listening for agent: {self.agent_name}")
    
    def stop_listening(self):
        """Stop message listening"""
        self._listening = False
        logger.info(f"Stopped message listening for agent: {self.agent_name}")
    
    def broadcast_message(self, message_type: str, content: Any, priority: str = "normal") -> bool:
        """Broadcast message to all agents"""
        return self.send_message("all", message_type, content, priority)
    
    def update_shared_data(self, key: str, value: Any) -> bool:
        """Update shared data accessible by all agents"""
        try:
            data = self._read_file(self.progress_file)
            data.setdefault("shared_data", {})[key] = value
            data["system_status"]["last_update"] = datetime.now().isoformat()
            
            return self._write_file(self.progress_file, data)
            
        except Exception as e:
            logger.error(f"Error updating shared data: {e}")
            return False
    
    def get_shared_data(self, key: str = None) -> Any:
        """Get shared data"""
        try:
            data = self._read_file(self.progress_file)
            shared_data = data.get("shared_data", {})
            
            if key:
                return shared_data.get(key)
            return shared_data
            
        except Exception as e:
            logger.error(f"Error getting shared data: {e}")
            return None
    
    def add_alert(self, level: str, message: str, category: str = "general") -> bool:
        """Add system alert"""
        try:
            alert = {
                "id": f"alert_{int(time.time() * 1000)}",
                "level": level,  # info, warning, error, critical
                "message": message,
                "category": category,
                "source": self.agent_name,
                "timestamp": datetime.now().isoformat(),
                "acknowledged": False
            }
            
            data = self._read_file(self.progress_file)
            alerts = data.setdefault("shared_data", {}).setdefault("system_alerts", [])
            alerts.append(alert)
            
            # Keep only last 100 alerts
            data["shared_data"]["system_alerts"] = alerts[-100:]
            
            return self._write_file(self.progress_file, data)
            
        except Exception as e:
            logger.error(f"Error adding alert: {e}")
            return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        try:
            data = self._read_file(self.progress_file)
            
            agents = data.get("agents", {})
            total_agents = len(agents)
            active_agents = sum(1 for agent in agents.values() 
                              if agent.get("status") in ["online", "busy"])
            
            # Calculate system health
            if active_agents == 0:
                health = "critical"
            elif active_agents < total_agents * 0.5:
                health = "degraded"
            elif active_agents < total_agents:
                health = "warning"
            else:
                health = "healthy"
            
            return {
                "health": health,
                "total_agents": total_agents,
                "active_agents": active_agents,
                "last_update": data.get("system_status", {}).get("last_update"),
                "alerts_count": len(data.get("shared_data", {}).get("system_alerts", []))
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {"health": "unknown", "error": str(e)}
    
    def cleanup_old_data(self, max_age_hours: int = 24) -> bool:
        """Cleanup old messages and logs"""
        try:
            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
            
            # Clean messages
            messages_data = self._read_file(self.messages_file)
            
            history = messages_data.get("message_history", [])
            filtered_history = []
            
            for msg in history:
                try:
                    msg_time = datetime.fromisoformat(msg["timestamp"]).timestamp()
                    if msg_time > cutoff_time:
                        filtered_history.append(msg)
                except (ValueError, KeyError):
                    # Keep message if timestamp parsing fails
                    filtered_history.append(msg)
            
            messages_data["message_history"] = filtered_history
            
            # Clean progress logs
            progress_data = self._read_file(self.progress_file)
            comm_log = progress_data.get("communication_log", [])
            filtered_log = []
            
            for log_entry in comm_log:
                try:
                    log_time = datetime.fromisoformat(log_entry["timestamp"]).timestamp()
                    if log_time > cutoff_time:
                        filtered_log.append(log_entry)
                except (ValueError, KeyError):
                    filtered_log.append(log_entry)
            
            progress_data["communication_log"] = filtered_log
            
            # Write cleaned data
            success1 = self._write_file(self.messages_file, messages_data)
            success2 = self._write_file(self.progress_file, progress_data)
            
            logger.info(f"Cleaned up old data (max age: {max_age_hours}h)")
            return success1 and success2
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return False