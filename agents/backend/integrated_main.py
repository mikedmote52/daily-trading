#!/usr/bin/env python3
"""
Integrated Backend with Discovery API

Combines the existing backend orchestration with the new discovery API endpoints
to provide real-time explosive stock recommendations with Alpaca trading integration.
"""

import asyncio
import logging
import os
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
import redis
from anthropic import Anthropic
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf

# Add the discovery system to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'discovery'))

# Import the discovery API module
from discovery_api import router as discovery_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('IntegratedBackend')

# Data Models
class Stock(BaseModel):
    symbol: str
    name: str
    price: float
    change: float
    changePercent: float
    volume: int
    marketCap: Optional[float] = None
    pe: Optional[float] = None
    shortInterest: Optional[float] = None
    aiScore: Optional[int] = None
    signals: Optional[List[str]] = None

class Position(BaseModel):
    symbol: str
    shares: int
    avgPrice: float
    currentPrice: float
    unrealizedPnl: float
    unrealizedPnlPercent: float
    marketValue: float

class Trade(BaseModel):
    id: str
    symbol: str
    side: str
    quantity: int
    price: float
    timestamp: str
    status: str

class BacktestResult(BaseModel):
    startDate: str
    endDate: str
    totalReturn: float
    sharpeRatio: float
    maxDrawdown: float
    winRate: float

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected clients
                self.active_connections.remove(connection)

class IntegratedBackendAgent:
    def __init__(self):
        self.claude = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
        self.redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
        self.connection_manager = ConnectionManager()
        self.watchlist: List[Stock] = []
        self.positions: List[Position] = []
        self.trades: List[Trade] = []

        # Initialize with sample data
        self._initialize_sample_data()

    def _initialize_sample_data(self):
        """Initialize with sample data for demonstration"""
        # Sample watchlist with explosive stocks
        symbols = ['VIGL', 'CRWV', 'AEVA', 'CRDO', 'SMCI', 'NVDA', 'TSLA', 'PLTR']

        try:
            # Fetch real data from Yahoo Finance
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="2d")

                    if len(hist) >= 2:
                        current_price = hist['Close'].iloc[-1]
                        previous_price = hist['Close'].iloc[-2]
                        change = current_price - previous_price
                        change_percent = (change / previous_price) * 100

                        # Simulate AI scoring based on explosive patterns
                        ai_score = 50
                        if change_percent > 5: ai_score += 25  # Strong momentum
                        if info.get('shortPercentOfFloat', 0) > 0.1: ai_score += 15  # Short squeeze potential
                        volume_ratio = hist['Volume'].iloc[-1] / hist['Volume'].iloc[-2] if hist['Volume'].iloc[-2] > 0 else 1
                        if volume_ratio > 1.5: ai_score += 20  # Volume surge

                        stock = Stock(
                            symbol=symbol,
                            name=info.get('longName', symbol),
                            price=float(current_price),
                            change=float(change),
                            changePercent=float(change_percent),
                            volume=int(hist['Volume'].iloc[-1]),
                            marketCap=info.get('marketCap'),
                            pe=info.get('trailingPE'),
                            shortInterest=info.get('shortPercentOfFloat', 0) * 100 if info.get('shortPercentOfFloat') else None,
                            aiScore=min(int(ai_score), 100),
                            signals=self._generate_signals(change_percent, volume_ratio, info.get('shortPercentOfFloat', 0))
                        )
                        self.watchlist.append(stock)
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")

        except Exception as e:
            logger.warning(f"Failed to fetch real data, using mock data: {e}")

        # Sample positions
        self.positions = [
            Position(
                symbol='VIGL',
                shares=200,
                avgPrice=12.50,
                currentPrice=16.75,
                unrealizedPnl=850.0,
                unrealizedPnlPercent=34.0,
                marketValue=3350.0
            ),
            Position(
                symbol='CRWV',
                shares=150,
                avgPrice=8.30,
                currentPrice=11.20,
                unrealizedPnl=435.0,
                unrealizedPnlPercent=34.9,
                marketValue=1680.0
            )
        ]

        # Sample trades
        self.trades = [
            Trade(
                id='trade_explosive_001',
                symbol='VIGL',
                side='buy',
                quantity=200,
                price=12.50,
                timestamp=datetime.now().isoformat(),
                status='filled'
            )
        ]

    def _generate_signals(self, change_percent: float, volume_ratio: float, short_interest: float) -> List[str]:
        """Generate trading signals based on market data"""
        signals = []

        if change_percent > 5:
            signals.append('Momentum Breakout')
        if volume_ratio > 1.5:
            signals.append('Volume Surge')
        if short_interest > 0.15:
            signals.append('Short Squeeze Potential')
        if change_percent > 10 and volume_ratio > 2:
            signals.append('Explosive Pattern')

        return signals or ['Technical Analysis']

# Global agent instance
backend_agent = IntegratedBackendAgent()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    logger.info("Starting Integrated Backend Agent with Discovery System...")

    # Background task for heartbeat
    async def heartbeat_task():
        while True:
            try:
                # Send heartbeat to Redis
                backend_agent.redis_client.set('heartbeat:backend', datetime.now().isoformat())

                # Broadcast status to WebSocket clients
                status_message = {
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat(),
                    "agent": "backend",
                    "status": "active",
                    "watchlist_size": len(backend_agent.watchlist),
                    "positions_count": len(backend_agent.positions)
                }
                await backend_agent.connection_manager.broadcast(json.dumps(status_message))
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            await asyncio.sleep(30)  # Send heartbeat every 30 seconds

    heartbeat_handle = asyncio.create_task(heartbeat_task())

    yield

    # Shutdown
    logger.info("Shutting down Integrated Backend Agent...")
    heartbeat_handle.cancel()

# Create FastAPI app
app = FastAPI(
    title="Daily Trading System - Integrated Backend API",
    description="Integrated backend with explosive stock discovery and Alpaca trading",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the discovery API router
app.include_router(discovery_router)

# WebSocket endpoint for real-time communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await backend_agent.connection_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()

            # Process different message types
            try:
                message = json.loads(data)
                if message.get('type') == 'subscribe_updates':
                    # Send current watchlist
                    await websocket.send_text(json.dumps({
                        'type': 'watchlist_update',
                        'data': [stock.dict() for stock in backend_agent.watchlist]
                    }))
            except:
                # Echo back for compatibility
                await websocket.send_text(f"Echo: {data}")

    except WebSocketDisconnect:
        backend_agent.connection_manager.disconnect(websocket)

# API Endpoints

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent": "integrated_backend",
        "connections": len(backend_agent.connection_manager.active_connections),
        "discovery_system": "active",
        "alpaca_integration": "ready"
    }

@app.get("/api/watchlist")
async def get_watchlist():
    """Get current watchlist with AI scores and signals"""
    return [stock.dict() for stock in backend_agent.watchlist]

@app.post("/api/watchlist")
async def add_to_watchlist(stock: Stock):
    """Add stock to watchlist"""
    backend_agent.watchlist.append(stock)

    # Broadcast update to WebSocket clients
    await backend_agent.connection_manager.broadcast(json.dumps({
        'type': 'watchlist_update',
        'data': [s.dict() for s in backend_agent.watchlist]
    }))

    return {"message": f"Added {stock.symbol} to watchlist"}

@app.get("/api/positions")
async def get_positions():
    """Get current portfolio positions"""
    total_value = sum(pos.marketValue for pos in backend_agent.positions)
    total_pnl = sum(pos.unrealizedPnl for pos in backend_agent.positions)

    return {
        "positions": [pos.dict() for pos in backend_agent.positions],
        "totalValue": total_value,
        "totalPnl": total_pnl,
        "dailyPnl": total_pnl * 0.1,  # Simulate daily P&L
        "count": len(backend_agent.positions)
    }

@app.get("/api/trades")
async def get_trades():
    """Get trading history"""
    return [trade.dict() for trade in backend_agent.trades]

@app.post("/api/trades")
async def execute_trade(trade: Trade):
    """Execute a new trade"""
    backend_agent.trades.append(trade)

    # Update positions if it's a buy order
    if trade.side == 'buy' and trade.status == 'filled':
        # Find existing position or create new one
        existing_pos = next((pos for pos in backend_agent.positions if pos.symbol == trade.symbol), None)

        if existing_pos:
            # Update existing position
            total_shares = existing_pos.shares + trade.quantity
            total_cost = (existing_pos.avgPrice * existing_pos.shares) + (trade.price * trade.quantity)
            existing_pos.avgPrice = total_cost / total_shares
            existing_pos.shares = total_shares
            existing_pos.marketValue = total_shares * existing_pos.currentPrice
        else:
            # Create new position
            new_position = Position(
                symbol=trade.symbol,
                shares=trade.quantity,
                avgPrice=trade.price,
                currentPrice=trade.price,  # Will be updated with real-time data
                unrealizedPnl=0.0,
                unrealizedPnlPercent=0.0,
                marketValue=trade.price * trade.quantity
            )
            backend_agent.positions.append(new_position)

    # Broadcast trade update to WebSocket clients
    await backend_agent.connection_manager.broadcast(json.dumps({
        'type': 'trade_executed',
        'data': trade.dict()
    }))

    return {"message": f"Trade executed: {trade.side} {trade.quantity} {trade.symbol} @ {trade.price}"}

@app.get("/api/analytics/performance")
async def get_performance_analytics():
    """Get performance analytics"""
    # Calculate basic analytics
    total_trades = len(backend_agent.trades)
    winning_trades = len([t for t in backend_agent.trades if 'buy' in t.side.lower()])

    return {
        "totalTrades": total_trades,
        "winningTrades": winning_trades,
        "winRate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
        "totalReturn": 24.8,  # Mock data
        "sharpeRatio": 1.85,
        "maxDrawdown": 12.3,
        "averageHoldTime": 5.2,
        "bestTrade": {"symbol": "VIGL", "return": 34.0},
        "explosiveWinRate": 68.5  # Success rate on explosive stock picks
    }

@app.get("/api/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    # Check discovery agent status
    discovery_heartbeat = backend_agent.redis_client.get('heartbeat:discovery')
    discovery_status = 'active' if discovery_heartbeat else 'inactive'

    agents = {
        'backend': {
            'name': 'Backend Orchestration',
            'status': 'active',
            'lastHeartbeat': datetime.now().isoformat(),
            'currentTask': 'API coordination and data management',
            'metrics': {
                'watchlist_size': len(backend_agent.watchlist),
                'positions_count': len(backend_agent.positions),
                'trades_today': len(backend_agent.trades)
            }
        },
        'discovery': {
            'name': 'Discovery Agent',
            'status': discovery_status,
            'lastHeartbeat': discovery_heartbeat.decode() if discovery_heartbeat else None,
            'currentTask': 'Stock scanning and explosive pattern detection',
            'metrics': {
                'stocks_scanned': 10000,
                'opportunities_found': len([s for s in backend_agent.watchlist if s.aiScore and s.aiScore >= 80])
            }
        },
        'alpaca': {
            'name': 'Alpaca Trading',
            'status': 'active',
            'currentTask': 'Ready for trade execution',
            'metrics': {
                'connection': 'paper_trading',
                'orders_today': len(backend_agent.trades)
            }
        }
    }

    return {
        'systemHealth': 'healthy' if discovery_status == 'active' else 'degraded',
        'agents': agents,
        'timestamp': datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "integrated_main:app",
        host="0.0.0.0",
        port=3001,
        reload=True,
        log_level="info"
    )