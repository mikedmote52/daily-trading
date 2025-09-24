#!/usr/bin/env python3
"""
Portfolio API Service
FastAPI service to expose portfolio management functionality
"""

from fastapi import FastAPI, HTTPException, Request
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
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PortfolioAPI")

app = FastAPI(
    title="AlphaStack Portfolio API",
    description="Real-time portfolio management and analytics",
    version="1.0.1"
)

# CORS middleware - Allow all origins to fix CORS issues
ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting storage
request_timestamps = {}
RATE_LIMIT_REQUESTS = 15  # requests per minute (higher for portfolio data)
RATE_LIMIT_WINDOW = 60    # seconds

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware to prevent API abuse"""
    # Skip rate limiting for health checks
    if request.url.path == "/health":
        return await call_next(request)

    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()

    # Clean old timestamps
    if client_ip in request_timestamps:
        request_timestamps[client_ip] = [
            timestamp for timestamp in request_timestamps[client_ip]
            if current_time - timestamp < RATE_LIMIT_WINDOW
        ]
    else:
        request_timestamps[client_ip] = []

    # Check rate limit
    if len(request_timestamps[client_ip]) >= RATE_LIMIT_REQUESTS:
        logger.warning(f"⚠️ Rate limit exceeded for {client_ip}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds."
        )

    # Add current timestamp
    request_timestamps[client_ip].append(current_time)

    # Process request
    response = await call_next(request)

    # Add rate limit headers
    remaining = RATE_LIMIT_REQUESTS - len(request_timestamps[client_ip])
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(int(current_time + RATE_LIMIT_WINDOW))

    return response

# Environment configuration - EXACT match from working orders service
ALPACA_BASE = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_KEY = os.environ.get("ALPACA_KEY", "")  # Exact same as working orders API
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")  # Exact same as working orders API

def _auth_headers():
    """Get Alpaca authentication headers"""
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json"
    }

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
    """Enhanced health check endpoint with dependency validation"""
    health_status = {
        "status": "healthy",
        "service": "portfolio-api",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "dependencies": {}
    }

    alpaca_configured = bool(ALPACA_KEY and ALPACA_SECRET)

    # Test Alpaca connection with detailed diagnostics
    alpaca_connected = False
    connection_error = None
    account_balance = None

    if alpaca_configured:
        try:
            start_time = time.time()
            with httpx.Client(base_url=ALPACA_BASE, headers=_auth_headers()) as client:
                response = client.get("/v2/account", timeout=10.0)
                latency_ms = int((time.time() - start_time) * 1000)

                if response.status_code == 200:
                    alpaca_connected = True
                    account_data = response.json()
                    account_balance = account_data.get("portfolio_value", "unknown")
                    health_status["dependencies"]["alpaca"] = {
                        "status": "connected",
                        "latency_ms": latency_ms,
                        "account_balance": account_balance
                    }
                else:
                    connection_error = f"HTTP {response.status_code}: {response.text[:100]}"
                    health_status["dependencies"]["alpaca"] = {
                        "status": "authentication_failed",
                        "error": connection_error,
                        "latency_ms": latency_ms
                    }
                    health_status["status"] = "unhealthy"
        except httpx.TimeoutException:
            connection_error = "Connection timeout"
            health_status["dependencies"]["alpaca"] = {"status": "timeout", "error": connection_error}
            health_status["status"] = "degraded"
        except Exception as e:
            connection_error = str(e)[:100]
            health_status["dependencies"]["alpaca"] = {"status": "failed", "error": connection_error}
            health_status["status"] = "unhealthy"
    else:
        health_status["dependencies"]["alpaca"] = {"status": "not_configured"}
        health_status["status"] = "unhealthy"

    # Environment variables check
    health_status["dependencies"]["environment"] = {
        "alpaca_key": bool(ALPACA_KEY),
        "alpaca_secret": bool(ALPACA_SECRET),
        "alpaca_base_url": bool(ALPACA_BASE),
        "alpaca_key_length": len(ALPACA_KEY) if ALPACA_KEY else 0
    }

    # Legacy fields for backwards compatibility
    health_status.update({
        "alpaca_configured": alpaca_configured,
        "alpaca_connected": alpaca_connected,
        "alpaca_base": ALPACA_BASE,
        "alpaca_key_present": bool(ALPACA_KEY),
        "alpaca_secret_present": bool(ALPACA_SECRET),
        "alpaca_key_length": len(ALPACA_KEY) if ALPACA_KEY else 0,
        "connection_error": connection_error
    })

    return health_status

@app.get("/portfolio", response_model=PortfolioSummary)
def get_portfolio_summary():
    """Get portfolio summary from Alpaca"""
    try:
        with httpx.Client(base_url=ALPACA_BASE, headers=_auth_headers()) as client:
            response = client.get("/v2/account")

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Alpaca API error: {response.text}"
                )

            account = response.json()

            # Calculate daily P&L from positions
            daily_pnl = 0.0
            positions_response = client.get("/v2/positions")

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
        with httpx.Client(base_url=ALPACA_BASE, headers=_auth_headers(), timeout=10) as client:
            response = client.get("/v2/positions")

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Alpaca API error: {response.text}"
                )

            positions = response.json()

            # Get portfolio value for weight calculation
            account_response = client.get("/v2/account")

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
        with httpx.Client(base_url=ALPACA_BASE, headers=_auth_headers()) as client:
            # Get portfolio history for performance calculation
            response = client.get(
                "/v2/account/portfolio/history",
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

@app.get("/enhanced-positions")
def get_enhanced_positions():
    """Get positions enriched with discovery context and current market data"""
    try:
        # Get current positions from Alpaca
        base_positions = get_positions()

        # Get discovery API to fetch current market intelligence
        discovery_base = "https://alphastack-discovery.onrender.com"
        enhanced_positions = []

        with httpx.Client(timeout=10) as client:
            for position in base_positions:
                symbol = position.symbol
                enhanced_pos = position.dict()

                try:
                    # Get current market data for the symbol via discovery system
                    # This gives us current RVOL, accumulation score, and signals
                    discovery_response = client.get(f"{discovery_base}/signals/current/{symbol}")

                    if discovery_response.status_code == 200:
                        discovery_data = discovery_response.json()
                        enhanced_pos.update({
                            "current_rvol": discovery_data.get("rvol", "N/A"),
                            "current_score": discovery_data.get("accumulation_score", "N/A"),
                            "current_signals": discovery_data.get("signals", []),
                            "discovery_context": {
                                "status": "ACTIVE" if discovery_data.get("rvol", 0) > 1.3 else "DORMANT",
                                "strength": "HIGH" if discovery_data.get("accumulation_score", 0) > 70 else "MODERATE"
                            }
                        })
                    else:
                        # Fallback - position not in current discovery results
                        enhanced_pos.update({
                            "current_rvol": "N/A",
                            "current_score": "N/A",
                            "current_signals": [],
                            "discovery_context": {
                                "status": "NOT_TRACKED",
                                "strength": "UNKNOWN"
                            }
                        })

                except Exception as e:
                    logger.warning(f"Failed to get discovery data for {symbol}: {e}")
                    enhanced_pos.update({
                        "current_rvol": "ERROR",
                        "current_score": "ERROR",
                        "current_signals": [],
                        "discovery_context": {
                            "status": "ERROR",
                            "strength": "UNKNOWN"
                        }
                    })

                # Add risk management recommendations
                unrealized_pl_pct = position.unrealized_plpc * 100
                enhanced_pos["risk_assessment"] = {
                    "stop_loss_suggestion": position.avg_entry_price * 0.90,  # 10% stop loss
                    "take_profit_suggestion": position.avg_entry_price * 1.20,  # 20% take profit
                    "risk_level": "HIGH" if unrealized_pl_pct < -15 else ("MODERATE" if unrealized_pl_pct < -5 else "LOW"),
                    "action_needed": unrealized_pl_pct < -20  # Flag major losses
                }

                # Add time-based insights
                enhanced_pos["insights"] = []
                if unrealized_pl_pct < -10:
                    enhanced_pos["insights"].append(f"Position down {unrealized_pl_pct:.1f}% - consider stop loss")
                if enhanced_pos.get("current_rvol", 0) and enhanced_pos["current_rvol"] != "N/A" and float(enhanced_pos["current_rvol"]) > 2.0:
                    enhanced_pos["insights"].append("High volume activity detected - monitor closely")
                if enhanced_pos["discovery_context"]["status"] == "DORMANT":
                    enhanced_pos["insights"].append("Discovery signals weakening - reassess position")

                enhanced_positions.append(enhanced_pos)

        return enhanced_positions

    except Exception as e:
        logger.error(f"Error getting enhanced positions: {e}")
        # Fallback to basic positions if enhancement fails
        return get_positions()

@app.get("/recommendations")
def get_ai_recommendations():
    """Get AI-powered portfolio recommendations based on current positions and discovery system"""
    try:
        positions = get_positions()

        if not positions:
            return {
                "recommendations": [
                    {
                        "action": "DISCOVER",
                        "message": "No current positions. Use discovery system to find explosive opportunities.",
                        "confidence": 95,
                        "timestamp": datetime.now().isoformat()
                    }
                ],
                "risk_assessment": {
                    "level": "LOW",
                    "factors": ["No positions", "No market exposure"],
                    "suggestions": ["Start with small position sizes from discovery system"]
                }
            }

        # Analyze current positions
        total_loss_positions = sum(1 for p in positions if p.unrealized_pl < 0)
        avg_loss_pct = sum(p.unrealized_plpc for p in positions if p.unrealized_plpc < 0) / max(total_loss_positions, 1) * 100

        recommendations = []
        risk_factors = []
        suggestions = []

        # Position-specific recommendations
        for position in positions:
            loss_pct = position.unrealized_plpc * 100
            if loss_pct < -20:
                recommendations.append({
                    "action": "URGENT_REVIEW",
                    "message": f"{position.symbol} down {loss_pct:.1f}% - immediate attention needed",
                    "confidence": 90,
                    "symbol": position.symbol
                })
            elif loss_pct < -10:
                recommendations.append({
                    "action": "MONITOR",
                    "message": f"{position.symbol} down {loss_pct:.1f}% - consider stop loss",
                    "confidence": 75,
                    "symbol": position.symbol
                })

        # Overall portfolio recommendations
        if total_loss_positions > len(positions) * 0.7:  # >70% losing positions
            recommendations.append({
                "action": "REASSESS",
                "message": "High percentage of losing positions - review discovery system effectiveness",
                "confidence": 85
            })
            risk_factors.append("High loss rate across portfolio")
            suggestions.append("Reduce position sizes until discovery accuracy improves")

        # Risk level determination
        if avg_loss_pct < -15:
            risk_level = "HIGH"
            suggestions.append("Consider implementing stricter stop losses")
        elif avg_loss_pct < -5:
            risk_level = "MODERATE"
            suggestions.append("Monitor positions closely for reversal signals")
        else:
            risk_level = "LOW"
            suggestions.append("Continue current strategy with position monitoring")

        if not recommendations:
            recommendations.append({
                "action": "HOLD",
                "message": "Portfolio performing within acceptable parameters",
                "confidence": 80,
                "timestamp": datetime.now().isoformat()
            })

        return {
            "recommendations": recommendations,
            "risk_assessment": {
                "level": risk_level,
                "factors": risk_factors or ["Normal market volatility"],
                "suggestions": suggestions
            },
            "portfolio_stats": {
                "total_positions": len(positions),
                "losing_positions": total_loss_positions,
                "avg_loss_pct": round(avg_loss_pct, 2) if total_loss_positions > 0 else 0
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

@app.get("/portfolio/enhanced")
def portfolio_enhanced():
    """Enhanced portfolio positions with discovery context"""
    return get_enhanced_positions()

@app.get("/portfolio/alerts")
def portfolio_alerts():
    """Portfolio alerts based on current positions and market conditions"""
    try:
        positions = get_positions()
        alerts = []
        alert_id = 1

        # Generate alerts based on position performance
        for position in positions:
            loss_pct = position.unrealized_plpc * 100

            if loss_pct < -20:
                alerts.append({
                    "id": alert_id,
                    "type": "CRITICAL",
                    "symbol": position.symbol,
                    "message": f"{position.symbol} down {loss_pct:.1f}% - urgent review required",
                    "timestamp": datetime.now().isoformat(),
                    "action_required": True
                })
                alert_id += 1

            elif loss_pct < -10:
                alerts.append({
                    "id": alert_id,
                    "type": "WARNING",
                    "symbol": position.symbol,
                    "message": f"{position.symbol} down {loss_pct:.1f}% - monitor closely",
                    "timestamp": datetime.now().isoformat(),
                    "action_required": False
                })
                alert_id += 1

        # Add system status alert
        if not alerts:
            alerts.append({
                "id": alert_id,
                "type": "INFO",
                "message": "All positions performing within normal parameters",
                "timestamp": datetime.now().isoformat(),
                "action_required": False
            })

        return {"alerts": alerts}

    except Exception as e:
        logger.error(f"Error generating alerts: {e}")
        return {
            "alerts": [
                {
                    "id": 1,
                    "type": "ERROR",
                    "message": f"Error generating alerts: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "action_required": True
                }
            ]
        }

@app.get("/test-auth")
def test_auth_endpoint():
    """Test endpoint to debug authentication - same pattern as orders service"""
    try:
        # Show key/secret previews for debugging
        key_preview = f"{ALPACA_KEY[:4]}...{ALPACA_KEY[-4:]}" if len(ALPACA_KEY) > 8 else ALPACA_KEY
        secret_preview = f"{ALPACA_SECRET[:4]}...{ALPACA_SECRET[-4:]}" if len(ALPACA_SECRET) > 8 else ALPACA_SECRET

        # Get the exact headers that will be sent
        headers = _auth_headers()

        # Test both account and positions endpoints
        results = {}

        with httpx.Client(base_url=ALPACA_BASE, headers=headers, timeout=10) as client:
            # Test account endpoint (like orders service)
            account_response = client.get("/v2/account")
            results["account"] = {
                "status_code": account_response.status_code,
                "response_text": account_response.text[:200]
            }

            # Test positions endpoint
            positions_response = client.get("/v2/positions")
            results["positions"] = {
                "status_code": positions_response.status_code,
                "response_text": positions_response.text[:200]
            }

        return {
            "alpaca_base": ALPACA_BASE,
            "key_length": len(ALPACA_KEY),
            "secret_length": len(ALPACA_SECRET),
            "key_preview": key_preview,
            "secret_preview": secret_preview,
            "key_has_whitespace": ALPACA_KEY != ALPACA_KEY.strip(),
            "secret_has_whitespace": ALPACA_SECRET != ALPACA_SECRET.strip(),
            "stripped_key_preview": f"{ALPACA_KEY.strip()[:4]}...{ALPACA_KEY.strip()[-4:]}",
            "stripped_secret_preview": f"{ALPACA_SECRET.strip()[:4]}...{ALPACA_SECRET.strip()[-4:]}",
            "headers_preview": {k: v[:10] + "..." if len(v) > 10 else v for k, v in headers.items()},
            "test_results": results
        }

    except Exception as e:
        logger.error(f"Test auth error: {e}")
        return {"error": str(e)}

@app.get("/test-orders-pattern")
def test_orders_pattern():
    """Test using the exact same pattern as the working orders service"""
    try:
        # EXACT same environment variable access as orders service
        alpaca_base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        alpaca_key = os.environ.get("ALPACA_KEY", "")
        alpaca_secret = os.environ.get("ALPACA_SECRET", "")

        # Debug: Show exact values being used
        debug_info = {
            "raw_key": repr(alpaca_key),
            "raw_secret": repr(alpaca_secret),
            "key_length": len(alpaca_key),
            "secret_length": len(alpaca_secret),
            "key_has_whitespace": alpaca_key != alpaca_key.strip(),
            "secret_has_whitespace": alpaca_secret != alpaca_secret.strip(),
            "key_stripped": alpaca_key.strip(),
            "secret_stripped": alpaca_secret.strip()
        }

        # EXACT same headers function as orders service with stripped values
        def orders_auth_headers():
            return {
                "APCA-API-KEY-ID": alpaca_key.strip(),
                "APCA-API-SECRET-KEY": alpaca_secret.strip(),
                "Content-Type": "application/json"
            }

        # EXACT same pattern as orders service account endpoint
        with httpx.Client(base_url=alpaca_base, headers=orders_auth_headers(), timeout=10) as client:
            response = client.get("/v2/account")

            if response.status_code >= 300:
                return {
                    "status": "FAILED",
                    "status_code": response.status_code,
                    "response": response.text[:200],
                    "key_preview": f"{alpaca_key[:4]}...{alpaca_key[-4:]}" if len(alpaca_key) > 8 else alpaca_key,
                    "debug": debug_info
                }

            account_data = response.json()

        return {
            "status": "SUCCESS",
            "account_id": account_data.get("id"),
            "buying_power": account_data.get("buying_power"),
            "cash": account_data.get("cash"),
            "portfolio_value": account_data.get("portfolio_value"),
            "key_preview": f"{alpaca_key[:4]}...{alpaca_key[-4:]}" if len(alpaca_key) > 8 else alpaca_key,
            "debug": debug_info
        }

    except Exception as e:
        return {"status": "ERROR", "error": str(e), "debug": debug_info if 'debug_info' in locals() else None}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    host = "0.0.0.0"
    print(f"Starting Portfolio API on {host}:{port}")
    print(f"Environment check - ALPACA_KEY: {len(os.environ.get('ALPACA_KEY', ''))} chars")
    print(f"Environment check - ALPACA_SECRET: {len(os.environ.get('ALPACA_SECRET', ''))} chars")
    uvicorn.run(app, host=host, port=port)