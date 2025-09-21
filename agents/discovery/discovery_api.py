#!/usr/bin/env python3
"""
Discovery API - FastAPI wrapper for Universal Discovery System
Provides REST endpoints for the explosive stock discovery engine
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
import os
import json
from datetime import datetime
from typing import List, Dict, Any
import redis
from contextlib import asynccontextmanager

# Import the universal discovery system
from universal_discovery import UniversalDiscoverySystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DiscoveryAPI")

# Global discovery system instance
discovery_system = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global discovery_system, redis_client

    logger.info("🚀 Initializing Explosive Discovery API...")

    # Initialize Redis connection
    try:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.warning(f"⚠️  Redis connection failed: {e}")
        redis_client = None

    # Initialize discovery system
    try:
        discovery_system = UniversalDiscoverySystem()
        logger.info("✅ Discovery API ready for deployment")
    except Exception as e:
        logger.error(f"❌ Failed to initialize discovery system: {e}")
        raise

    yield

    # Cleanup
    if redis_client:
        redis_client.close()

# FastAPI app with lifespan
app = FastAPI(
    title="AlphaStack Discovery API",
    description="AI-powered explosive stock discovery engine",
    version="2.0.1",
    lifespan=lifespan
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

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "discovery-api",
        "version": "2.0.1",
        "discovery_ready": discovery_system is not None,
        "redis_connected": redis_client is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/discover")
async def run_discovery():
    """Run the discovery pipeline and return top candidates"""
    if not discovery_system:
        raise HTTPException(status_code=503, detail="Discovery system not initialized")

    try:
        # Generate unique scan ID
        scan_id = f"scan_{int(datetime.now().timestamp())}"
        logger.info(f"🔍 Starting discovery scan: {scan_id}")

        # Run discovery
        results = discovery_system.run_discovery()

        # Cache results if Redis is available
        if redis_client:
            try:
                redis_client.setex(f"discovery:latest", 300, json.dumps(results))
                redis_client.setex(f"discovery:{scan_id}", 3600, json.dumps(results))
            except Exception as e:
                logger.warning(f"⚠️  Redis cache failed: {e}")

        logger.info(f"✅ Discovery scan complete: {scan_id}")

        return {
            "scan_id": scan_id,
            "timestamp": datetime.now().isoformat(),
            "candidates": results
        }

    except Exception as e:
        logger.error(f"❌ Discovery scan failed: {e}")
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")

@app.get("/signals/top")
async def get_top_signals():
    """Get top discovery signals"""
    if not discovery_system:
        raise HTTPException(status_code=503, detail="Discovery system not initialized")

    try:
        logger.info("🚀 RUNNING REAL DISCOVERY PIPELINE")

        # Run discovery
        results = discovery_system.run_discovery()

        # Format for frontend
        formatted_results = []
        for candidate in results:
            formatted_results.append({
                "symbol": candidate["symbol"],
                "price": candidate["last_price"],
                "score": candidate["accumulation_score"],
                "volume": candidate["day_volume"],
                "rvol": candidate.get("rvol_sust", 0),
                "reason": candidate.get("accumulation_reason", "Strong accumulation pattern detected")
            })

        return formatted_results

    except Exception as e:
        logger.error(f"❌ Signals request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Signals failed: {str(e)}")

@app.get("/")
async def root():
    """API documentation"""
    return {
        "service": "AlphaStack Discovery API",
        "description": "AI-powered explosive stock discovery engine",
        "version": "2.0.1",
        "endpoints": {
            "POST /discover": "Run discovery pipeline",
            "GET /signals/top": "Get top discovery signals",
            "GET /health": "Health check"
        },
        "status": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    logger.info(f"🚀 Starting Discovery API on {host}:{port}")
    uvicorn.run(app, host=host, port=port, workers=1)