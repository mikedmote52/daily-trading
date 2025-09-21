#!/usr/bin/env python3
"""
Portfolio API Service
FastAPI service to expose portfolio management functionality
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PortfolioAPI")

app = FastAPI(
    title="AlphaStack Portfolio API",
    description="Real-time portfolio management and analytics",
    version="1.0.0"
)

# CORS middleware
ALLOWED_ORIGINS = [
    "https://alphastack-frontend.onrender.com",
    "http://localhost:5173",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment configuration
ALPACA_BASE = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_KEY = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")

def _auth_headers():
    """Get Alpaca authentication headers"""
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json"
    }
    logger.info(f"Auth headers: APCA-API-KEY-ID={ALPACA_KEY[:4]}...{ALPACA_KEY[-4:] if len(ALPACA_KEY) > 8 else ''}")
    logger.info(f"ALPACA_BASE: {ALPACA_BASE}")
    return headers

class PortfolioSummary(BaseModel):
    portfolio_value: float
    cash: float
    daily_pnl: float
    daily_pnl_percent: float
    positions_count: int
    buying_power: float
    equity: float

class Position(BaseModel):
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    unrealized_pl: float
    unrealized_plpc: float
    current_price: float
    weight: float

class PerformanceMetrics(BaseModel):
    total_return: float
    total_return_percent: float
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]
    volatility: Optional[float]
    win_rate: Optional[float]

@app.get("/health")
async def health():
    """Health check endpoint"""
    alpaca_configured = bool(ALPACA_KEY and ALPACA_SECRET)

    # Test Alpaca connection
    alpaca_connected = False
    connection_error = None

    if alpaca_configured:
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{ALPACA_BASE}/v2/account",
                    headers=_auth_headers(),
                    timeout=5.0
                )
                alpaca_connected = response.status_code == 200
                if not alpaca_connected:
                    connection_error = f"HTTP {response.status_code}: {response.text[:100]}"
        except Exception as e:
            alpaca_connected = False
            connection_error = str(e)[:100]

    return {
        "status": "healthy",
        "service": "portfolio-api",
        "alpaca_configured": alpaca_configured,
        "alpaca_connected": alpaca_connected,
        "alpaca_base": ALPACA_BASE,
        "alpaca_key_present": bool(ALPACA_KEY),
        "alpaca_secret_present": bool(ALPACA_SECRET),
        "alpaca_key_length": len(ALPACA_KEY) if ALPACA_KEY else 0,
        "connection_error": connection_error,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/portfolio", response_model=PortfolioSummary)
def get_portfolio_summary():
    """Get portfolio summary from Alpaca"""
    try:
        with httpx.Client() as client:
            response = client.get(
                f"{ALPACA_BASE}/v2/account",
                headers=_auth_headers()
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Alpaca API error: {response.text}"
                )

            account = response.json()

            # Calculate daily P&L from positions
            daily_pnl = 0.0
            positions_response = client.get(
                f"{ALPACA_BASE}/v2/positions",
                headers=_auth_headers()
            )

            if positions_response.status_code == 200:
                positions = positions_response.json()
                daily_pnl = sum(float(pos.get('unrealized_pl', 0)) for pos in positions)

            portfolio_value = float(account['portfolio_value'])
            daily_pnl_percent = (daily_pnl / portfolio_value * 100) if portfolio_value > 0 else 0

            return PortfolioSummary(
                portfolio_value=portfolio_value,
                cash=float(account['cash']),
                daily_pnl=daily_pnl,
                daily_pnl_percent=daily_pnl_percent,
                positions_count=len(positions) if positions_response.status_code == 200 else 0,
                buying_power=float(account['buying_power']),
                equity=float(account['equity'])
            )

    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Failed to connect to Alpaca: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positions", response_model=List[Position])
def get_positions():
    """Get current positions from Alpaca"""
    try:
        headers = _auth_headers()
        url = f"{ALPACA_BASE}/v2/positions"
        logger.info(f"Making request to: {url}")
        logger.info(f"Headers keys: {list(headers.keys())}")

        with httpx.Client() as client:
            response = client.get(url, headers=headers)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Alpaca API error: {response.text}"
                )

            positions = response.json()

            # Get portfolio value for weight calculation
            account_response = client.get(
                f"{ALPACA_BASE}/v2/account",
                headers=_auth_headers()
            )

            portfolio_value = 1.0  # Default to avoid division by zero
            if account_response.status_code == 200:
                account = account_response.json()
                portfolio_value = float(account['portfolio_value'])

            result = []
            for pos in positions:
                if float(pos['qty']) != 0:  # Only include non-zero positions
                    market_value = float(pos['market_value'])
                    qty = float(pos['qty'])
                    current_price = market_value / abs(qty) if qty != 0 else float(pos['avg_entry_price'])

                    result.append(Position(
                        symbol=pos['symbol'],
                        qty=qty,
                        market_value=market_value,
                        avg_entry_price=float(pos['avg_entry_price']),
                        unrealized_pl=float(pos['unrealized_pl']),
                        unrealized_plpc=float(pos['unrealized_plpc']),
                        current_price=current_price,
                        weight=market_value / portfolio_value if portfolio_value > 0 else 0
                    ))

            return result

    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Failed to connect to Alpaca: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance", response_model=PerformanceMetrics)
def get_performance():
    """Get portfolio performance metrics"""
    try:
        with httpx.Client() as client:
            # Get portfolio history for performance calculation
            response = client.get(
                f"{ALPACA_BASE}/v2/account/portfolio/history",
                headers=_auth_headers(),
                params={"period": "1M", "timeframe": "1D"}
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Alpaca API error: {response.text}"
                )

            history = response.json()

            if not history.get('equity') or len(history['equity']) < 2:
                # Not enough data for calculations
                return PerformanceMetrics(
                    total_return=0.0,
                    total_return_percent=0.0,
                    sharpe_ratio=None,
                    max_drawdown=None,
                    volatility=None,
                    win_rate=None
                )

            equity_values = [float(val) for val in history['equity'] if val is not None]

            if len(equity_values) < 2:
                return PerformanceMetrics(
                    total_return=0.0,
                    total_return_percent=0.0,
                    sharpe_ratio=None,
                    max_drawdown=None,
                    volatility=None,
                    win_rate=None
                )

            # Calculate metrics
            initial_value = equity_values[0]
            final_value = equity_values[-1]
            total_return = final_value - initial_value
            total_return_percent = (total_return / initial_value * 100) if initial_value > 0 else 0

            # Calculate daily returns for additional metrics
            daily_returns = []
            for i in range(1, len(equity_values)):
                if equity_values[i-1] > 0:
                    daily_return = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                    daily_returns.append(daily_return)

            # Calculate additional metrics if we have enough data
            sharpe_ratio = None
            max_drawdown = None
            volatility = None
            win_rate = None

            if len(daily_returns) > 7:  # At least a week of data
                import numpy as np

                daily_returns_array = np.array(daily_returns)

                # Volatility (annualized)
                volatility = float(np.std(daily_returns_array) * np.sqrt(252) * 100)

                # Sharpe ratio (assuming 2% risk-free rate)
                excess_returns = daily_returns_array - (0.02 / 252)
                if np.std(excess_returns) > 0:
                    sharpe_ratio = float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))

                # Max drawdown
                cumulative_returns = np.cumprod(1 + daily_returns_array)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdown = float(np.min(drawdowns) * 100)

                # Win rate
                winning_days = np.sum(daily_returns_array > 0)
                win_rate = float(winning_days / len(daily_returns_array) * 100)

            return PerformanceMetrics(
                total_return=total_return,
                total_return_percent=total_return_percent,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                win_rate=win_rate
            )

    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Failed to connect to Alpaca: {str(e)}")
    except ImportError:
        # numpy not available - return basic metrics only
        return PerformanceMetrics(
            total_return=0.0,
            total_return_percent=0.0,
            sharpe_ratio=None,
            max_drawdown=None,
            volatility=None,
            win_rate=None
        )
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations")
def get_ai_recommendations():
    """Get AI-powered portfolio recommendations"""
    try:
        # This would integrate with the discovery system and Claude AI
        # For now, return a placeholder
        return {
            "recommendations": [
                {
                    "action": "HOLD",
                    "message": "Portfolio is well-balanced. Continue monitoring market conditions.",
                    "confidence": 85,
                    "timestamp": datetime.now().isoformat()
                }
            ],
            "risk_assessment": {
                "level": "MODERATE",
                "factors": ["Market volatility", "Position concentration"],
                "suggestions": ["Consider rebalancing if any position exceeds 20%"]
            }
        }

    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Frontend-expected endpoints with /portfolio prefix
@app.get("/portfolio/health")
def portfolio_health():
    """Health check endpoint - frontend expects /portfolio/health"""
    return health()

@app.get("/portfolio/summary")
def portfolio_summary():
    """Portfolio summary - frontend expects /portfolio/summary"""
    return get_portfolio_summary()

@app.get("/portfolio/positions")
def portfolio_positions():
    """Portfolio positions - frontend expects /portfolio/positions"""
    return get_positions()

@app.get("/portfolio/recommendations")
def portfolio_recommendations():
    """Portfolio recommendations - frontend expects /portfolio/recommendations"""
    return get_ai_recommendations()

@app.get("/portfolio/alerts")
def portfolio_alerts():
    """Portfolio alerts - placeholder endpoint"""
    return {
        "alerts": [
            {
                "id": 1,
                "type": "INFO",
                "message": "Portfolio system is operational",
                "timestamp": datetime.now().isoformat()
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    host = "0.0.0.0"
    print(f"Starting Portfolio API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)