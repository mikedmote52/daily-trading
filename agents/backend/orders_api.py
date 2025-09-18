#!/usr/bin/env python3
"""
AlphaStack Orders API - Alpaca Paper Trading Integration
Handles bracket orders with stop-loss and take-profit for discovered stocks
Updated: 2025-09-18 - API credentials configured
"""
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
import logging
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OrdersAPI")

# FastAPI app
app = FastAPI(
    title="AlphaStack Orders API",
    description="Paper trading orders for explosive stock discovery",
    version="1.0.0"
)

# CORS middleware for frontend communication
ALLOWED_ORIGINS = [
    "https://alphastack-frontend.onrender.com",  # Render static URL
    "http://localhost:5173",                     # local dev
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
LIMIT_BAND = float(os.environ.get("LIMIT_BAND_PCT", "0.005"))  # 0.5% limit band
SL_PCT = float(os.environ.get("SL_PCT_DEFAULT", "0.10"))      # 10% stop loss
TP_PCT = float(os.environ.get("TP_PCT_DEFAULT", "0.20"))      # 20% take profit

def _auth_headers():
    """Get Alpaca authentication headers"""
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json"
    }

class OrderRequest(BaseModel):
    ticker: str
    notional_usd: float = 100.0
    last_price: float
    side: str = "buy"

@app.get("/health")
def health():
    """Health check for Render deployment"""
    return {
        "status": "healthy",
        "service": "orders-api",
        "paper_trading": True,
        "alpaca_configured": bool(ALPACA_KEY and ALPACA_SECRET),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
def root():
    """API documentation"""
    return {
        "service": "AlphaStack Orders API",
        "description": "Paper trading orders for explosive stock discovery",
        "endpoints": {
            "POST /orders": "Create bracket order (buy + stop-loss + take-profit)",
            "GET /health": "Health check",
            "GET /account": "Account information",
            "GET /positions": "Current positions"
        },
        "paper_trading": True,
        "alpaca_base": ALPACA_BASE
    }

@app.post("/orders")
def create_order(
    payload: OrderRequest,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key")
):
    """
    Create paper bracket order for discovered stock

    Bracket order includes:
    1. Main buy order (limit price with small buffer)
    2. Stop-loss order (protect downside)
    3. Take-profit order (capture upside)
    """
    if not idempotency_key:
        raise HTTPException(status_code=400, detail="Idempotency-Key header required")

    if not ALPACA_KEY or not ALPACA_SECRET:
        raise HTTPException(status_code=500, detail="Alpaca credentials not configured")

    ticker = payload.ticker.upper()
    notional = payload.notional_usd
    last_price = payload.last_price

    if last_price <= 0:
        raise HTTPException(status_code=400, detail="last_price must be > 0")

    # Calculate order prices
    limit_price = round(last_price * (1 + LIMIT_BAND), 2)  # Small buffer above market
    sl_price = round(last_price * (1 - SL_PCT), 2)         # Stop loss below entry
    tp_price = round(last_price * (1 + TP_PCT), 2)         # Take profit above entry
    qty = max(1, int(notional / last_price))               # Calculate shares

    # Construct bracket order
    order_payload = {
        "symbol": ticker,
        "qty": str(qty),
        "side": payload.side,
        "type": "limit",
        "time_in_force": "day",
        "limit_price": str(limit_price),
        "order_class": "bracket",
        "take_profit": {
            "limit_price": str(tp_price)
        },
        "stop_loss": {
            "stop_price": str(sl_price),
            "limit_price": str(round(sl_price * 0.95, 2))  # Stop-limit for better fills
        }
    }

    logger.info(f"Creating bracket order for {ticker}: Entry=${limit_price}, SL=${sl_price}, TP=${tp_price}")

    try:
        with httpx.Client(base_url=ALPACA_BASE, headers=_auth_headers(), timeout=15) as client:
            response = client.post("/v2/orders", json=order_payload)

            if response.status_code >= 300:
                logger.error(f"Alpaca order failed: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Alpaca API error: {response.text}"
                )

            alpaca_data = response.json()

        return {
            "intent_id": idempotency_key,
            "status": "accepted",
            "ticker": ticker,
            "qty": qty,
            "order_details": {
                "entry_price": limit_price,
                "stop_loss": sl_price,
                "take_profit": tp_price,
                "notional_usd": round(qty * last_price, 2)
            },
            "alpaca_response": alpaca_data,
            "timestamp": datetime.now().isoformat()
        }

    except httpx.RequestError as e:
        logger.error(f"Network error: {e}")
        raise HTTPException(status_code=503, detail="Network error connecting to Alpaca")
    except Exception as e:
        logger.error(f"Order creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Order creation failed: {str(e)}")

@app.get("/account")
def get_account():
    """Get Alpaca account information"""
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise HTTPException(status_code=500, detail="Alpaca credentials not configured")

    try:
        with httpx.Client(base_url=ALPACA_BASE, headers=_auth_headers(), timeout=10) as client:
            response = client.get("/v2/account")

            if response.status_code >= 300:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            account_data = response.json()

        return {
            "account_id": account_data.get("id"),
            "buying_power": account_data.get("buying_power"),
            "cash": account_data.get("cash"),
            "portfolio_value": account_data.get("portfolio_value"),
            "day_trade_count": account_data.get("day_trade_count"),
            "pattern_day_trader": account_data.get("pattern_day_trader"),
            "trade_suspended_by_user": account_data.get("trade_suspended_by_user"),
            "trading_blocked": account_data.get("trading_blocked"),
            "account_blocked": account_data.get("account_blocked"),
            "timestamp": datetime.now().isoformat()
        }

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail="Network error connecting to Alpaca")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Account fetch failed: {str(e)}")

@app.get("/positions")
def get_positions():
    """Get current positions"""
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise HTTPException(status_code=500, detail="Alpaca credentials not configured")

    try:
        with httpx.Client(base_url=ALPACA_BASE, headers=_auth_headers(), timeout=10) as client:
            response = client.get("/v2/positions")

            if response.status_code >= 300:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            positions = response.json()

        return {
            "positions": positions,
            "count": len(positions),
            "timestamp": datetime.now().isoformat()
        }

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail="Network error connecting to Alpaca")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Positions fetch failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "orders_api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        reload=True,
        log_level="info"
    )