#!/usr/bin/env python3
"""
Backend Orchestration Agent

Central API coordinator for the multi-agent trading system.
Handles communication between agents, data flow, and external API integrations.
"""

import asyncio
import logging
import os
import json
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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BackendAgent')

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
    side: str  # 'buy' or 'sell'
    quantity: int
    price: float
    timestamp: str
    status: str  # 'pending', 'filled', 'cancelled'
    strategy: Optional[str] = None

class TradeCommand(BaseModel):
    command: str

class BacktestRequest(BaseModel):
    strategy: str
    parameters: Dict[str, Any]

class BacktestResult(BaseModel):
    strategy: str
    totalReturn: float
    sharpeRatio: float
    maxDrawdown: float
    winRate: float
    totalTrades: int
    period: str

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected clients
                self.active_connections.remove(connection)

class BackendOrchestrationAgent:
    def __init__(self):
        self.claude = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
        self.redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
        self.connection_manager = ConnectionManager()
        self.watchlist: List[Stock] = []
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.backtest_results: List[BacktestResult] = []
        
        # Initialize with sample data
        self._initialize_sample_data()

    def _initialize_sample_data(self):
        """Initialize with sample data for demonstration"""
        # Sample watchlist
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX']
        
        try:
            # Fetch real data from Yahoo Finance
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="2d")
                
                if len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    previous_price = hist['Close'].iloc[-2]
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100
                    
                    stock = Stock(
                        symbol=symbol,
                        name=info.get('longName', symbol),
                        price=current_price,
                        change=change,
                        changePercent=change_percent,
                        volume=int(hist['Volume'].iloc[-1]),
                        marketCap=info.get('marketCap'),
                        pe=info.get('trailingPE'),
                        shortInterest=info.get('shortPercentOfFloat', 0) * 100 if info.get('shortPercentOfFloat') else None,
                        aiScore=85 + (hash(symbol) % 15),  # Mock AI score
                        signals=['Momentum', 'Volume Surge'] if change > 0 else ['Support Level']
                    )
                    self.watchlist.append(stock)
                    
        except Exception as e:
            logger.warning(f"Failed to fetch real data, using mock data: {e}")
            # Fallback to mock data
            for symbol in symbols:
                stock = Stock(
                    symbol=symbol,
                    name=f"{symbol} Inc.",
                    price=150.0 + (hash(symbol) % 100),
                    change=-2.5 + (hash(symbol) % 10),
                    changePercent=-1.5 + (hash(symbol) % 3),
                    volume=1000000 + (hash(symbol) % 500000),
                    marketCap=1000000000 + (hash(symbol) % 500000000),
                    pe=15.0 + (hash(symbol) % 20),
                    shortInterest=5.0 + (hash(symbol) % 15),
                    aiScore=70 + (hash(symbol) % 30),
                    signals=['Technical Analysis', 'Volume Pattern']
                )
                self.watchlist.append(stock)

        # Sample positions
        self.positions = [
            Position(
                symbol='AAPL',
                shares=100,
                avgPrice=145.50,
                currentPrice=self.watchlist[0].price if self.watchlist else 150.0,
                unrealizedPnl=450.0,
                unrealizedPnlPercent=3.1,
                marketValue=15000.0
            ),
            Position(
                symbol='GOOGL',
                shares=50,
                avgPrice=120.00,
                currentPrice=self.watchlist[1].price if len(self.watchlist) > 1 else 125.0,
                unrealizedPnl=250.0,
                unrealizedPnlPercent=4.2,
                marketValue=6250.0
            )
        ]

        # Sample trades
        self.trades = [
            Trade(
                id='trade_001',
                symbol='AAPL',
                side='buy',
                quantity=100,
                price=145.50,
                timestamp=(datetime.now() - timedelta(hours=2)).isoformat(),
                status='filled',
                strategy='AI Signals'
            ),
            Trade(
                id='trade_002',
                symbol='GOOGL',
                side='buy',
                quantity=50,
                price=120.00,
                timestamp=(datetime.now() - timedelta(hours=1)).isoformat(),
                status='filled',
                strategy='Momentum'
            )
        ]

        # Sample backtest results
        self.backtest_results = [
            BacktestResult(
                strategy='momentum',
                totalReturn=15.7,
                sharpeRatio=1.42,
                maxDrawdown=-8.3,
                winRate=62.5,
                totalTrades=120,
                period='2024-01-01 to 2024-12-31'
            ),
            BacktestResult(
                strategy='mean_reversion',
                totalReturn=8.9,
                sharpeRatio=0.89,
                maxDrawdown=-12.1,
                winRate=58.3,
                totalTrades=95,
                period='2024-01-01 to 2024-12-31'
            )
        ]

    async def send_heartbeat(self):
        """Send heartbeat to master agent"""
        try:
            heartbeat_data = {
                'type': 'heartbeat',
                'sender': 'backend',
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'active_connections': len(self.connection_manager.active_connections),
                    'watchlist_size': len(self.watchlist),
                    'positions_count': len(self.positions),
                    'trades_count': len(self.trades)
                }
            }
            
            self.redis_client.set('heartbeat:backend', datetime.now().isoformat())
            self.redis_client.publish('master_channel', json.dumps(heartbeat_data))
            
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")

    async def process_trade_command(self, command: str) -> Dict[str, Any]:
        """Process natural language trading commands using Claude"""
        try:
            prompt = f"""
            You are a trading assistant. Parse the following trading command and extract the trading parameters:
            
            Command: "{command}"
            
            Extract:
            1. Action (buy/sell)
            2. Symbol (stock ticker)
            3. Quantity (number of shares or dollar amount)
            4. Order type (market/limit)
            5. Price (if limit order)
            
            Respond with a JSON object containing these parameters.
            If the command is invalid or unclear, return an error message.
            
            Example response:
            {{
                "valid": true,
                "action": "buy",
                "symbol": "AAPL",
                "quantity": 100,
                "order_type": "market",
                "price": null
            }}
            """
            
            response = self.claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse Claude's response
            trade_params = json.loads(response.content[0].text)
            
            if trade_params.get('valid'):
                # Simulate trade execution
                trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Get current price for the symbol
                current_price = 150.0  # Default price
                for stock in self.watchlist:
                    if stock.symbol == trade_params['symbol']:
                        current_price = stock.price
                        break
                
                new_trade = Trade(
                    id=trade_id,
                    symbol=trade_params['symbol'],
                    side=trade_params['action'],
                    quantity=trade_params['quantity'],
                    price=trade_params.get('price', current_price),
                    timestamp=datetime.now().isoformat(),
                    status='filled',  # Simulate immediate fill for demo
                    strategy='Manual'
                )
                
                self.trades.insert(0, new_trade)
                
                return {
                    'success': True,
                    'trade': new_trade.dict(),
                    'message': f"Successfully executed {trade_params['action']} order for {trade_params['quantity']} shares of {trade_params['symbol']}"
                }
            else:
                return {
                    'success': False,
                    'error': trade_params.get('error', 'Invalid command format')
                }
                
        except Exception as e:
            logger.error(f"Error processing trade command: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def run_backtest(self, strategy: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run backtesting using Claude for analysis"""
        try:
            prompt = f"""
            Simulate running a backtest for the {strategy} strategy with these parameters:
            {json.dumps(parameters, indent=2)}
            
            Generate realistic performance metrics for a {strategy} strategy based on:
            - Historical market conditions
            - Strategy characteristics
            - Risk management rules
            
            Return a JSON object with:
            - totalReturn (percentage)
            - sharpeRatio
            - maxDrawdown (negative percentage)
            - winRate (percentage)
            - totalTrades (number)
            - period (date range)
            
            Make the results realistic and varied based on the strategy type.
            """
            
            response = self.claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse Claude's response
            backtest_data = json.loads(response.content[0].text)
            
            # Create BacktestResult object
            result = BacktestResult(
                strategy=strategy,
                totalReturn=backtest_data['totalReturn'],
                sharpeRatio=backtest_data['sharpeRatio'],
                maxDrawdown=backtest_data['maxDrawdown'],
                winRate=backtest_data['winRate'],
                totalTrades=backtest_data['totalTrades'],
                period=backtest_data['period']
            )
            
            # Add to results
            self.backtest_results.insert(0, result)
            
            return {
                'success': True,
                'result': result.dict(),
                'message': f"Backtest completed for {strategy} strategy"
            }
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Initialize the backend agent
backend_agent = BackendOrchestrationAgent()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Backend Orchestration Agent...")
    
    # Start background tasks
    async def heartbeat_task():
        while True:
            await backend_agent.send_heartbeat()
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
    
    heartbeat_handle = asyncio.create_task(heartbeat_task())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Backend Orchestration Agent...")
    heartbeat_handle.cancel()

# Create FastAPI app
app = FastAPI(
    title="Daily Trading System - Backend API",
    description="Backend orchestration agent for the multi-agent trading system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket endpoint for real-time communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await backend_agent.connection_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back for now - could process agent commands here
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
        "agent": "backend",
        "connections": len(backend_agent.connection_manager.active_connections)
    }

@app.get("/api/watchlist", response_model=List[Stock])
async def get_watchlist():
    """Get current watchlist"""
    return backend_agent.watchlist

@app.get("/api/positions")
async def get_positions():
    """Get current positions and portfolio summary"""
    total_value = sum(pos.marketValue for pos in backend_agent.positions)
    daily_pnl = sum(pos.unrealizedPnl for pos in backend_agent.positions)
    
    return {
        "positions": backend_agent.positions,
        "totalValue": total_value,
        "dailyPnl": daily_pnl
    }

@app.get("/api/trades", response_model=List[Trade])
async def get_trades():
    """Get recent trades"""
    return backend_agent.trades

@app.post("/api/trades/execute")
async def execute_trade(trade_command: TradeCommand):
    """Execute a trade using natural language command"""
    result = await backend_agent.process_trade_command(trade_command.command)
    
    if result['success']:
        # Broadcast to connected clients
        await backend_agent.connection_manager.broadcast(
            json.dumps({
                "type": "trade_executed",
                "data": result['trade']
            })
        )
        return result
    else:
        raise HTTPException(status_code=400, detail=result['error'])

@app.post("/api/backtest/run")
async def run_backtest(request: BacktestRequest):
    """Run a backtest"""
    result = await backend_agent.run_backtest(request.strategy, request.parameters)
    
    if result['success']:
        # Broadcast to connected clients
        await backend_agent.connection_manager.broadcast(
            json.dumps({
                "type": "backtest_completed",
                "data": result['result']
            })
        )
        return result
    else:
        raise HTTPException(status_code=400, detail=result['error'])

@app.get("/api/backtest/results", response_model=List[BacktestResult])
async def get_backtest_results():
    """Get backtest results"""
    return backend_agent.backtest_results

@app.get("/api/system/metrics")
async def get_system_metrics():
    """Get system-wide metrics"""
    return {
        "totalAgents": 6,  # Total number of agents in the system
        "activeAgents": 4,  # Number of currently active agents
        "totalTrades": len(backend_agent.trades),
        "portfolioValue": sum(pos.marketValue for pos in backend_agent.positions),
        "dailyPnl": sum(pos.unrealizedPnl for pos in backend_agent.positions),
        "systemHealth": "healthy"
    }

@app.get("/api/agents/status")
async def get_agent_status():
    """Get status of all agents"""
    try:
        registry_data = backend_agent.redis_client.get('agent_registry')
        if registry_data:
            return json.loads(registry_data)
        else:
            return {}
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        return {}

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=3001,
        reload=True,
        log_level="info"
    )