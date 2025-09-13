#!/usr/bin/env python3
"""
Redis Helper Utilities

Common Redis operations for all agents.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import redis
from dataclasses import asdict, is_dataclass

logger = logging.getLogger(__name__)

class RedisHelper:
    """Helper class for Redis operations"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.client = redis.Redis.from_url(redis_url)
        self._test_connection()
    
    def _test_connection(self):
        """Test Redis connection"""
        try:
            self.client.ping()
            logger.info("Redis connection established")
        except redis.ConnectionError:
            logger.error("Failed to connect to Redis")
            raise
    
    def serialize_data(self, data: Any) -> str:
        """Serialize data for Redis storage"""
        def default_serializer(obj):
            if is_dataclass(obj):
                return asdict(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)
        
        return json.dumps(data, default=default_serializer)
    
    def deserialize_data(self, data: str) -> Any:
        """Deserialize data from Redis"""
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
    
    def set_data(self, key: str, data: Any, expiry: Optional[int] = None) -> bool:
        """Set data in Redis with optional expiry"""
        try:
            serialized_data = self.serialize_data(data)
            self.client.set(key, serialized_data, ex=expiry)
            return True
        except Exception as e:
            logger.error(f"Error setting data for key {key}: {e}")
            return False
    
    def get_data(self, key: str) -> Optional[Any]:
        """Get data from Redis"""
        try:
            data = self.client.get(key)
            if data:
                return self.deserialize_data(data.decode())
            return None
        except Exception as e:
            logger.error(f"Error getting data for key {key}: {e}")
            return None
    
    def delete_data(self, key: str) -> bool:
        """Delete data from Redis"""
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting data for key {key}: {e}")
            return False
    
    def publish_message(self, channel: str, message: Dict[str, Any]) -> bool:
        """Publish message to Redis channel"""
        try:
            serialized_message = self.serialize_data(message)
            self.client.publish(channel, serialized_message)
            return True
        except Exception as e:
            logger.error(f"Error publishing message to channel {channel}: {e}")
            return False
    
    def subscribe_to_channel(self, channel: str):
        """Subscribe to Redis channel"""
        try:
            pubsub = self.client.pubsub()
            pubsub.subscribe(channel)
            return pubsub
        except Exception as e:
            logger.error(f"Error subscribing to channel {channel}: {e}")
            return None
    
    def set_heartbeat(self, agent_name: str) -> bool:
        """Set agent heartbeat"""
        return self.set_data(f"heartbeat:{agent_name}", datetime.now().isoformat(), 60)
    
    def get_heartbeat(self, agent_name: str) -> Optional[datetime]:
        """Get agent heartbeat"""
        heartbeat_data = self.get_data(f"heartbeat:{agent_name}")
        if heartbeat_data:
            try:
                return datetime.fromisoformat(heartbeat_data)
            except ValueError:
                pass
        return None
    
    def is_agent_alive(self, agent_name: str, timeout_seconds: int = 60) -> bool:
        """Check if agent is alive based on heartbeat"""
        heartbeat = self.get_heartbeat(agent_name)
        if heartbeat:
            time_diff = (datetime.now() - heartbeat).total_seconds()
            return time_diff <= timeout_seconds
        return False
    
    def set_agent_status(self, agent_name: str, status: Dict[str, Any]) -> bool:
        """Set comprehensive agent status"""
        status_data = {
            'agent': agent_name,
            'timestamp': datetime.now().isoformat(),
            **status
        }
        return self.set_data(f"agent_status:{agent_name}", status_data, 300)
    
    def get_agent_status(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get agent status"""
        return self.get_data(f"agent_status:{agent_name}")
    
    def get_all_agent_statuses(self) -> Dict[str, Any]:
        """Get all agent statuses"""
        statuses = {}
        pattern = "agent_status:*"
        
        try:
            keys = self.client.keys(pattern)
            for key in keys:
                agent_name = key.decode().replace("agent_status:", "")
                status = self.get_agent_status(agent_name)
                if status:
                    statuses[agent_name] = status
        except Exception as e:
            logger.error(f"Error getting all agent statuses: {e}")
        
        return statuses
    
    def cache_market_data(self, symbol: str, data: Dict[str, Any], expiry: int = 300) -> bool:
        """Cache market data with expiry"""
        return self.set_data(f"market_data:{symbol}", data, expiry)
    
    def get_cached_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached market data"""
        return self.get_data(f"market_data:{symbol}")
    
    def store_trading_signal(self, signal: Dict[str, Any]) -> bool:
        """Store trading signal"""
        signal_key = f"trading_signal:{signal.get('symbol', 'unknown')}:{datetime.now().timestamp()}"
        return self.set_data(signal_key, signal, 3600)  # 1 hour expiry
    
    def get_recent_signals(self, symbol: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trading signals"""
        pattern = f"trading_signal:{symbol}:*" if symbol else "trading_signal:*"
        signals = []
        
        try:
            keys = self.client.keys(pattern)
            # Sort by timestamp (descending)
            keys.sort(reverse=True)
            
            for key in keys[:limit]:
                signal = self.get_data(key.decode())
                if signal:
                    signals.append(signal)
        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
        
        return signals
    
    def store_backtest_result(self, result: Dict[str, Any]) -> bool:
        """Store backtest result"""
        result_key = f"backtest_result:{result.get('strategy', 'unknown')}:{datetime.now().timestamp()}"
        return self.set_data(result_key, result, 86400)  # 24 hour expiry
    
    def get_backtest_results(self, strategy: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get backtest results"""
        pattern = f"backtest_result:{strategy}:*" if strategy else "backtest_result:*"
        results = []
        
        try:
            keys = self.client.keys(pattern)
            keys.sort(reverse=True)
            
            for key in keys[:limit]:
                result = self.get_data(key.decode())
                if result:
                    results.append(result)
        except Exception as e:
            logger.error(f"Error getting backtest results: {e}")
        
        return results
    
    def cleanup_expired_data(self) -> int:
        """Cleanup expired data and return count of deleted keys"""
        deleted_count = 0
        
        try:
            # This is a simplified cleanup - Redis handles TTL automatically
            # But we can clean up old data that doesn't have TTL
            
            # Clean up old heartbeats (older than 5 minutes)
            cutoff_time = datetime.now().timestamp() - 300
            
            for pattern in ["heartbeat:*", "agent_status:*"]:
                keys = self.client.keys(pattern)
                for key in keys:
                    key_str = key.decode()
                    # Check if key should be cleaned up based on timestamp
                    data = self.get_data(key_str)
                    if data and isinstance(data, dict) and 'timestamp' in data:
                        try:
                            timestamp = datetime.fromisoformat(data['timestamp']).timestamp()
                            if timestamp < cutoff_time:
                                self.delete_data(key_str)
                                deleted_count += 1
                        except (ValueError, TypeError):
                            pass
            
            logger.info(f"Cleaned up {deleted_count} expired entries")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        return deleted_count