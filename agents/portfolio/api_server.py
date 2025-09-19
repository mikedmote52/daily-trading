#!/usr/bin/env python3
"""
Portfolio Management API Server

FastAPI server that provides portfolio management endpoints for the frontend.
Serves real-time portfolio data, health metrics, and AI recommendations.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from enhanced_portfolio_manager import EnhancedPortfolioManager

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PortfolioAPI")

# Global portfolio manager
portfolio_manager: Optional[EnhancedPortfolioManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage portfolio manager lifecycle"""
    global portfolio_manager
    
    # Startup
    logger.info("Starting Portfolio Management API...")
    try:
        portfolio_manager = EnhancedPortfolioManager()
        # Start background monitoring
        asyncio.create_task(portfolio_manager.start())
        logger.info("Portfolio manager started successfully")
    except Exception as e:
        logger.error(f"Failed to start portfolio manager: {e}")
        portfolio_manager = None
    
    yield
    
    # Shutdown
    if portfolio_manager:
        portfolio_manager.running = False
        logger.info("Portfolio manager stopped")

# Create FastAPI app
app = FastAPI(
    title="Enhanced Portfolio Management API",
    description="AI-powered portfolio management for explosive stock returns",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PortfolioSummary(BaseModel):
    total_value: float
    total_pnl: float
    total_pnl_percent: float
    cash: float
    position_count: int
    top_performer: Optional[Dict] = None
    worst_performer: Optional[Dict] = None
    overall_health: float

class PositionDetail(BaseModel):
    symbol: str
    shares: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    weight: float
    days_held: int
    technical_health: float
    fundamental_health: float
    thesis_health: float
    overall_health: float
    thesis_performance: str
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    max_drawdown: float

class RecommendationDetail(BaseModel):
    symbol: str
    action: str
    confidence: float
    rationale: str
    urgency: str
    suggested_shares: Optional[int] = None
    target_weight: Optional[float] = None
    risk_factors: List[str] = []

class PortfolioHealthDetail(BaseModel):
    total_value: float
    daily_pnl: float
    daily_pnl_percent: float
    total_pnl: float
    total_pnl_percent: float
    concentration_risk: float
    sector_diversification: float
    volatility_score: float
    correlation_risk: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    avg_winner: float
    avg_loser: float
    technical_health: float
    fundamental_health: float
    overall_health: float

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if portfolio_manager is None:
        raise HTTPException(status_code=503, detail="Portfolio manager not available")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "portfolio-management-api",
        "version": "1.0.0"
    }

@app.get("/portfolio/summary", response_model=PortfolioSummary)
async def get_portfolio_summary():
    """Get portfolio summary"""
    if portfolio_manager is None:
        raise HTTPException(status_code=503, detail="Portfolio manager not available")
    
    try:
        # Get account info from Alpaca
        account_info = portfolio_manager.alpaca_client.get_account_info()
        
        if not portfolio_manager.positions:
            return PortfolioSummary(
                total_value=account_info.get('portfolio_value', 0),
                total_pnl=0,
                total_pnl_percent=0,
                cash=account_info.get('cash', 0),
                position_count=0,
                overall_health=100
            )
        
        # Calculate metrics
        positions = list(portfolio_manager.positions.values())
        total_pnl = sum(pos.unrealized_pnl for pos in positions)
        total_cost = sum(pos.shares * pos.avg_price for pos in positions)
        total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        # Find best and worst performers
        best_performer = max(positions, key=lambda p: p.unrealized_pnl_percent)
        worst_performer = min(positions, key=lambda p: p.unrealized_pnl_percent)
        
        overall_health = sum(pos.overall_health for pos in positions) / len(positions)
        
        return PortfolioSummary(
            total_value=account_info.get('portfolio_value', 0),
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            cash=account_info.get('cash', 0),
            position_count=len(positions),
            top_performer={
                "symbol": best_performer.symbol,
                "pnl_percent": best_performer.unrealized_pnl_percent
            },
            worst_performer={
                "symbol": worst_performer.symbol,
                "pnl_percent": worst_performer.unrealized_pnl_percent
            },
            overall_health=overall_health
        )
        
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get portfolio summary")

@app.get("/portfolio/positions", response_model=List[PositionDetail])
async def get_portfolio_positions():
    """Get all portfolio positions with health metrics"""
    if portfolio_manager is None:
        raise HTTPException(status_code=503, detail="Portfolio manager not available")
    
    try:
        positions = []
        for pos in portfolio_manager.positions.values():
            positions.append(PositionDetail(
                symbol=pos.symbol,
                shares=pos.shares,
                avg_price=pos.avg_price,
                current_price=pos.current_price,
                market_value=pos.market_value,
                unrealized_pnl=pos.unrealized_pnl,
                unrealized_pnl_percent=pos.unrealized_pnl_percent,
                weight=pos.weight,
                days_held=pos.days_held,
                technical_health=pos.technical_health,
                fundamental_health=pos.fundamental_health,
                thesis_health=pos.thesis_health,
                overall_health=pos.overall_health,
                thesis_performance=pos.thesis_performance,
                price_target=pos.price_target,
                stop_loss=pos.stop_loss,
                max_drawdown=pos.max_drawdown
            ))
        
        # Sort by market value (largest positions first)
        positions.sort(key=lambda p: p.market_value, reverse=True)
        return positions
        
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get positions")

@app.get("/portfolio/position/{symbol}", response_model=PositionDetail)
async def get_position_detail(symbol: str):
    """Get detailed information for a specific position"""
    if portfolio_manager is None:
        raise HTTPException(status_code=503, detail="Portfolio manager not available")
    
    if symbol not in portfolio_manager.positions:
        raise HTTPException(status_code=404, detail=f"Position {symbol} not found")
    
    try:
        pos = portfolio_manager.positions[symbol]
        return PositionDetail(
            symbol=pos.symbol,
            shares=pos.shares,
            avg_price=pos.avg_price,
            current_price=pos.current_price,
            market_value=pos.market_value,
            unrealized_pnl=pos.unrealized_pnl,
            unrealized_pnl_percent=pos.unrealized_pnl_percent,
            weight=pos.weight,
            days_held=pos.days_held,
            technical_health=pos.technical_health,
            fundamental_health=pos.fundamental_health,
            thesis_health=pos.thesis_health,
            overall_health=pos.overall_health,
            thesis_performance=pos.thesis_performance,
            price_target=pos.price_target,
            stop_loss=pos.stop_loss,
            max_drawdown=pos.max_drawdown
        )
        
    except Exception as e:
        logger.error(f"Error getting position detail for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get position detail for {symbol}")

@app.get("/portfolio/recommendations", response_model=List[RecommendationDetail])
async def get_portfolio_recommendations():
    """Get AI-powered recommendations for all positions"""
    if portfolio_manager is None:
        raise HTTPException(status_code=503, detail="Portfolio manager not available")
    
    try:
        recommendations = []
        for rec in portfolio_manager.recommendations.values():
            recommendations.append(RecommendationDetail(
                symbol=rec.symbol,
                action=rec.action,
                confidence=rec.confidence,
                rationale=rec.rationale,
                urgency=rec.urgency,
                suggested_shares=rec.suggested_shares,
                target_weight=rec.target_weight,
                risk_factors=rec.risk_factors or []
            ))
        
        # Sort by urgency and confidence
        urgency_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        recommendations.sort(
            key=lambda r: (urgency_order.get(r.urgency, 0), r.confidence),
            reverse=True
        )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendations")

@app.get("/portfolio/recommendation/{symbol}", response_model=RecommendationDetail)
async def get_position_recommendation(symbol: str):
    """Get AI recommendation for a specific position"""
    if portfolio_manager is None:
        raise HTTPException(status_code=503, detail="Portfolio manager not available")
    
    if symbol not in portfolio_manager.recommendations:
        raise HTTPException(status_code=404, detail=f"No recommendation for {symbol}")
    
    try:
        rec = portfolio_manager.recommendations[symbol]
        return RecommendationDetail(
            symbol=rec.symbol,
            action=rec.action,
            confidence=rec.confidence,
            rationale=rec.rationale,
            urgency=rec.urgency,
            suggested_shares=rec.suggested_shares,
            target_weight=rec.target_weight,
            risk_factors=rec.risk_factors or []
        )
        
    except Exception as e:
        logger.error(f"Error getting recommendation for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendation for {symbol}")

@app.get("/portfolio/health", response_model=PortfolioHealthDetail)
async def get_portfolio_health():
    """Get comprehensive portfolio health metrics"""
    if portfolio_manager is None:
        raise HTTPException(status_code=503, detail="Portfolio manager not available")
    
    if portfolio_manager.portfolio_health is None:
        raise HTTPException(status_code=404, detail="Portfolio health data not available")
    
    try:
        health = portfolio_manager.portfolio_health
        return PortfolioHealthDetail(
            total_value=health.total_value,
            daily_pnl=health.daily_pnl,
            daily_pnl_percent=health.daily_pnl_percent,
            total_pnl=health.total_pnl,
            total_pnl_percent=health.total_pnl_percent,
            concentration_risk=health.concentration_risk,
            sector_diversification=health.sector_diversification,
            volatility_score=health.volatility_score,
            correlation_risk=health.correlation_risk,
            sharpe_ratio=health.sharpe_ratio,
            sortino_ratio=health.sortino_ratio,
            max_drawdown=health.max_drawdown,
            win_rate=health.win_rate,
            avg_winner=health.avg_winner,
            avg_loser=health.avg_loser,
            technical_health=health.technical_health,
            fundamental_health=health.fundamental_health,
            overall_health=health.overall_health
        )
        
    except Exception as e:
        logger.error(f"Error getting portfolio health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get portfolio health")

@app.post("/portfolio/refresh")
async def refresh_portfolio(background_tasks: BackgroundTasks):
    """Manually trigger portfolio refresh"""
    if portfolio_manager is None:
        raise HTTPException(status_code=503, detail="Portfolio manager not available")
    
    try:
        # Trigger immediate portfolio update
        background_tasks.add_task(portfolio_manager.update_portfolio)
        
        return {
            "status": "refresh_initiated",
            "timestamp": datetime.now().isoformat(),
            "message": "Portfolio refresh started in background"
        }
        
    except Exception as e:
        logger.error(f"Error refreshing portfolio: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh portfolio")

@app.get("/portfolio/performance")
async def get_portfolio_performance():
    """Get portfolio performance metrics and history"""
    if portfolio_manager is None:
        raise HTTPException(status_code=503, detail="Portfolio manager not available")
    
    try:
        # Get portfolio history from Alpaca
        history = portfolio_manager.alpaca_client.get_portfolio_history('1M')
        
        # Calculate performance metrics
        performance_data = {
            "current_status": {
                "total_positions": len(portfolio_manager.positions),
                "total_value": portfolio_manager.portfolio_health.total_value if portfolio_manager.portfolio_health else 0,
                "total_pnl": portfolio_manager.portfolio_health.total_pnl if portfolio_manager.portfolio_health else 0,
                "total_pnl_percent": portfolio_manager.portfolio_health.total_pnl_percent if portfolio_manager.portfolio_health else 0
            },
            "risk_metrics": {
                "concentration_risk": portfolio_manager.portfolio_health.concentration_risk if portfolio_manager.portfolio_health else 0,
                "max_drawdown": portfolio_manager.portfolio_health.max_drawdown if portfolio_manager.portfolio_health else 0,
                "win_rate": portfolio_manager.portfolio_health.win_rate if portfolio_manager.portfolio_health else 0
            },
            "historical_performance": history,
            "top_positions": [
                {
                    "symbol": pos.symbol,
                    "pnl_percent": pos.unrealized_pnl_percent,
                    "market_value": pos.market_value,
                    "health": pos.overall_health
                }
                for pos in sorted(portfolio_manager.positions.values(), key=lambda p: p.unrealized_pnl_percent, reverse=True)[:5]
            ]
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error getting portfolio performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get portfolio performance")

@app.get("/portfolio/alerts")
async def get_portfolio_alerts():
    """Get current portfolio alerts and warnings"""
    if portfolio_manager is None:
        raise HTTPException(status_code=503, detail="Portfolio manager not available")
    
    try:
        alerts = []
        
        # Check for critical recommendations
        for rec in portfolio_manager.recommendations.values():
            if rec.urgency in ["CRITICAL", "HIGH"]:
                alerts.append({
                    "type": "recommendation",
                    "symbol": rec.symbol,
                    "action": rec.action,
                    "urgency": rec.urgency,
                    "message": f"{rec.action} {rec.symbol}: {rec.rationale}",
                    "confidence": rec.confidence
                })
        
        # Check for health issues
        for pos in portfolio_manager.positions.values():
            if pos.overall_health < 30:
                alerts.append({
                    "type": "health_warning",
                    "symbol": pos.symbol,
                    "urgency": "HIGH",
                    "message": f"{pos.symbol} health critically low ({pos.overall_health:.1f}/100)",
                    "health_score": pos.overall_health
                })
            elif pos.unrealized_pnl_percent < -15:
                alerts.append({
                    "type": "loss_warning",
                    "symbol": pos.symbol,
                    "urgency": "MEDIUM",
                    "message": f"{pos.symbol} down {pos.unrealized_pnl_percent:.1f}% - consider stop loss",
                    "pnl_percent": pos.unrealized_pnl_percent
                })
        
        # Check portfolio-level risks
        if portfolio_manager.portfolio_health:
            health = portfolio_manager.portfolio_health
            if health.concentration_risk > 25:
                alerts.append({
                    "type": "concentration_risk",
                    "urgency": "MEDIUM",
                    "message": f"Portfolio concentration risk high ({health.concentration_risk:.1f}%)",
                    "concentration": health.concentration_risk
                })
        
        # Sort by urgency
        urgency_order = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
        alerts.sort(key=lambda a: urgency_order.get(a["urgency"], 0), reverse=True)
        
        return {
            "alerts": alerts,
            "total_alerts": len(alerts),
            "critical_count": len([a for a in alerts if a["urgency"] == "CRITICAL"]),
            "high_count": len([a for a in alerts if a["urgency"] == "HIGH"]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting portfolio alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get portfolio alerts")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
