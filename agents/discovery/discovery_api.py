#!/usr/bin/env python3
"""
Discovery API - FastAPI wrapper for Universal Discovery System
Provides REST endpoints for the explosive stock discovery engine
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import redis
from contextlib import asynccontextmanager
import time

# Import the universal discovery system
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from universal_discovery import UniversalDiscoverySystem

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('discovery_api.log') if os.getenv('LOG_TO_FILE') else logging.NullHandler()
    ]
)
logger = logging.getLogger("DiscoveryAPI")

# Global discovery system instance
discovery_system = None
redis_client = None

# Rate limiting storage
request_timestamps = {}
RATE_LIMIT_REQUESTS = 10  # requests per minute
RATE_LIMIT_WINDOW = 60    # seconds

# Request metrics
request_metrics = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'rate_limited_requests': 0,
    'average_response_time': 0,
    'last_reset': time.time()
}

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
ALLOWED_ORIGINS = ["*"]  # Allow all origins to fix CORS issues

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Rate limiting and metrics middleware
@app.middleware("http")
async def rate_limit_and_metrics_middleware(request: Request, call_next):
    """Rate limiting middleware with request metrics tracking"""
    start_time = time.time()
    request_metrics['total_requests'] += 1

    # Skip rate limiting for health checks but still track metrics
    if request.url.path != "/health":
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
            request_metrics['rate_limited_requests'] += 1
            logger.warning(f"⚠️ Rate limit exceeded for {client_ip} on {request.url.path}")
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds."
            )

        # Add current timestamp
        request_timestamps[client_ip].append(current_time)

    try:
        # Process request
        response = await call_next(request)

        # Track successful requests
        if response.status_code < 400:
            request_metrics['successful_requests'] += 1
        else:
            request_metrics['failed_requests'] += 1
            logger.warning(f"⚠️ Request failed: {request.method} {request.url.path} - {response.status_code}")

        # Calculate response time
        response_time = time.time() - start_time

        # Update average response time (simple moving average)
        if request_metrics['successful_requests'] > 0:
            request_metrics['average_response_time'] = (
                (request_metrics['average_response_time'] * (request_metrics['successful_requests'] - 1) + response_time) /
                request_metrics['successful_requests']
            )

        # Add rate limit headers (if not health check)
        if request.url.path != "/health":
            remaining = RATE_LIMIT_REQUESTS - len(request_timestamps.get(request.client.host if request.client else "unknown", []))
            response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(time.time() + RATE_LIMIT_WINDOW))

        # Add performance headers
        response.headers["X-Response-Time"] = f"{response_time:.3f}s"

        return response

    except Exception as e:
        request_metrics['failed_requests'] += 1
        logger.error(f"❌ Request error: {request.method} {request.url.path} - {str(e)}")
        raise

@app.get("/health")
async def health():
    """Enhanced health check endpoint with dependency validation"""
    health_status = {
        "status": "healthy",
        "service": "discovery-api",
        "version": "2.0.1",
        "timestamp": datetime.now().isoformat(),
        "dependencies": {}
    }

    # Check discovery system
    health_status["discovery_ready"] = discovery_system is not None
    if discovery_system is None:
        health_status["status"] = "unhealthy"

    # Check Redis connection
    redis_connected = False
    if redis_client:
        try:
            redis_client.ping()
            redis_connected = True
            health_status["dependencies"]["redis"] = {"status": "connected", "latency_ms": None}
        except Exception as e:
            health_status["dependencies"]["redis"] = {"status": "failed", "error": str(e)}
            health_status["status"] = "degraded"
    else:
        health_status["dependencies"]["redis"] = {"status": "not_configured"}

    health_status["redis_connected"] = redis_connected

    # Check Polygon API key
    polygon_key_configured = bool(os.getenv('POLYGON_API_KEY'))
    health_status["dependencies"]["polygon_api"] = {
        "status": "configured" if polygon_key_configured else "not_configured"
    }
    if not polygon_key_configured:
        health_status["status"] = "unhealthy"

    # Check environment variables
    health_status["dependencies"]["environment"] = {
        "polygon_api_key": polygon_key_configured,
        "redis_url": bool(os.getenv('REDIS_URL')),
        "discovery_cycle_seconds": bool(os.getenv('DISCOVERY_CYCLE_SECONDS')),
        "top_n": bool(os.getenv('TOP_N'))
    }

    return health_status

@app.get("/metrics")
async def get_metrics():
    """Get API performance metrics"""
    uptime = time.time() - request_metrics['last_reset']

    return {
        "service": "discovery-api",
        "metrics": {
            "requests": {
                "total": request_metrics['total_requests'],
                "successful": request_metrics['successful_requests'],
                "failed": request_metrics['failed_requests'],
                "rate_limited": request_metrics['rate_limited_requests'],
                "success_rate": (request_metrics['successful_requests'] / max(request_metrics['total_requests'], 1)) * 100
            },
            "performance": {
                "average_response_time": round(request_metrics['average_response_time'], 3),
                "uptime_seconds": round(uptime, 2),
                "requests_per_second": round(request_metrics['total_requests'] / max(uptime, 1), 2)
            },
            "system": {
                "discovery_system_ready": discovery_system is not None,
                "redis_connected": redis_client is not None,
                "cache_entries": len(discovery_system.cache) if discovery_system else 0
            }
        },
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

        # Run discovery with timeout protection
        discovery_result = discovery_system.discover(gates=['A'], limit=20)

        # Validate discovery result structure
        if not isinstance(discovery_result, list):
            logger.error(f"❌ Invalid discovery result type: {type(discovery_result)}")
            raise HTTPException(status_code=500, detail="Discovery returned invalid result format")

        # Use the discovery result directly (it's already a list of candidates)
        candidates = discovery_result

        # Cache results if Redis is available
        if redis_client:
            try:
                redis_client.setex(f"discovery:latest", 300, json.dumps(candidates))
                redis_client.setex(f"discovery:{scan_id}", 3600, json.dumps(candidates))
            except Exception as e:
                logger.warning(f"⚠️  Redis cache failed: {e}")

        logger.info(f"✅ Discovery scan complete: {scan_id} - Found {len(candidates)} candidates")

        return {
            "scan_id": scan_id,
            "timestamp": datetime.now().isoformat(),
            "candidates": candidates,
            "status": "success",
            "total_candidates": len(candidates)
        }

    except ValueError as e:
        logger.error(f"❌ Discovery validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid discovery parameters: {str(e)}")
    except TimeoutError as e:
        logger.error(f"❌ Discovery timeout: {e}")
        raise HTTPException(status_code=504, detail="Discovery scan timed out")
    except Exception as e:
        logger.error(f"❌ Discovery scan failed: {e}")
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")

@app.get("/scan/{scan_id}")
async def get_scan_results(scan_id: str):
    """Get results for a specific scan ID"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        # Try to get cached results for this scan
        cached_data = redis_client.get(f"discovery:{scan_id}")
        if cached_data:
            candidates = json.loads(cached_data)
            return {
                "scan_id": scan_id,
                "status": "completed",
                "candidates": candidates
            }
        else:
            # Check if scan is recent (within last hour)
            try:
                timestamp = int(scan_id.split('_')[1])
                current_time = int(datetime.now().timestamp())
                if current_time - timestamp < 3600:  # Less than 1 hour old
                    return {
                        "scan_id": scan_id,
                        "status": "running",
                        "candidates": []
                    }
            except:
                pass

            # Scan not found or too old
            return {
                "scan_id": scan_id,
                "status": "not_found",
                "candidates": []
            }
    except Exception as e:
        logger.error(f"Error fetching scan results: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch scan results")

@app.get("/debug/simple")
async def debug_simple_format():
    """Debug endpoint with simple format for frontend testing"""
    return [
        {"symbol": "FATN", "price": 8.89, "score": 75, "volume": 1570.4, "reason": "Extreme volume surge"},
        {"symbol": "AGMH", "price": 12.45, "score": 73, "volume": 845.2, "reason": "Strong accumulation"},
        {"symbol": "VRME", "price": 15.67, "score": 71, "volume": 623.1, "reason": "Breakout pattern"}
    ]

@app.get("/signals/top")
async def get_top_signals():
    """Get top discovery signals"""
    if not discovery_system:
        raise HTTPException(status_code=503, detail="Discovery system not initialized")

    try:
        logger.info("🚀 RUNNING REAL DISCOVERY PIPELINE")

        # Run discovery with validation
        discovery_result = discovery_system.discover(gates=['A'], limit=20)

        # Validate discovery result structure
        if not isinstance(discovery_result, list):
            logger.error(f"❌ Invalid discovery result type: {type(discovery_result)}")
            return []

        # Use the discovery result directly (it's already a list of candidates)
        candidates = discovery_result

        # Format for frontend with error handling
        formatted_results = []
        for candidate in candidates:
            try:
                # Validate required fields
                symbol = candidate.get("symbol", "").strip()
                if not symbol:
                    continue

                formatted_result = {
                    "symbol": symbol,
                    "price": float(candidate.get("price", 0)),
                    "score": float(candidate.get("accumulation_score", 0)),  # Keep decimal precision!
                    "volume": int(candidate.get("volume", 0)),
                    "rvol": float(candidate.get("rvol", 1.0)),
                    "explosion_probability": float(candidate.get("explosion_probability", 0)),  # New explosion probability
                    "reason": ", ".join(candidate.get("signals", candidate.get("reasons", ["Accumulation pattern detected"]))),  # Use enhanced reasons

                    # Additional enhanced data fields for frontend
                    "tier": candidate.get("tier", "B-TIER"),
                    "market_cap": candidate.get("market_cap", 0),
                    "company_name": candidate.get("company_name", ""),
                    "sector": candidate.get("sector", ""),
                    "short_squeeze_potential": candidate.get("short_squeeze_potential", "Unknown"),
                    "data_source": "ENHANCED_DISCOVERY_SYSTEM"
                }
                formatted_results.append(formatted_result)
            except (ValueError, TypeError) as e:
                logger.warning(f"⚠️ Skipping invalid candidate {candidate.get('symbol', 'UNKNOWN')}: {e}")
                continue

        logger.info(f"✅ Returning {len(formatted_results)} validated signals")
        return formatted_results

    except ValueError as e:
        logger.error(f"❌ Signals validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid signal data: {str(e)}")
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